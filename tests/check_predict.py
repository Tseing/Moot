import math
import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append("..")
from ..src.model.optformer import IncrementalDecoder, Transformer
from ..src.tokenizer import StrTokenizer


class Generator:
    def __init__(
        self,
        model: Transformer,
        tokenizer: StrTokenizer,
        maxlen: int,
        beam_size: int = 1,
        minlen: int = 1,
        stop_early: bool = True,
        normalize_scores: bool = True,
        len_penalty=1,
        unk_penalty: float = 0,
        retain_dropout: bool = False,
        sampling: bool = False,
        sampling_topk: int = -1,
        sampling_temperature=1,
    ):
        """Generates translations of a given source sentence.
        Args:
            min/maxlen: The length of the generated output will be bounded by
                minlen and maxlen (not including the end-of-sentence marker).
            stop_early: Stop generation immediately after we finalize beam_size
                hypotheses, even though longer hypotheses might have better
                normalized scores.
            normalize_scores: Normalize scores by the length of the output.
        """
        self.model = model
        self.pad = tokenizer.vocab2index[tokenizer.pad]
        self.unk = tokenizer.vocab2index[tokenizer.unk]
        self.eos = tokenizer.vocab2index[tokenizer.eos]
        self.vocab_size = tokenizer.vocab_size
        self.beam_size = beam_size

        self.minlen = minlen
        # max_decoder_len = min(m.max_decoder_positions() for m in self.models)
        # max_decoder_len -= 1  # we define maxlen not including the EOS marker
        # self.maxlen = max_decoder_len if maxlen is None else min(maxlen, max_decoder_len)
        self.maxlen = maxlen
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.sampling = sampling
        self.sampling_topk = sampling_topk
        self.sampling_temperature = sampling_temperature

    def generate(self, src_tokens, src_lengths, beam_size=None, maxlen=None, prefix_tokens=None):
        """Generate a batch of translations."""
        with torch.no_grad():
            return self._generate(src_tokens, src_lengths, beam_size, maxlen, prefix_tokens)

    def _generate(self, src_tokens, src_lengths, beam_size=None, maxlen=None, prefix_tokens=None):
        bsz, srclen = src_tokens.size()
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

        # the max beam size is the dictionary size - 1, since we never select pad
        beam_size = beam_size if beam_size is not None else self.beam_size
        beam_size = min(beam_size, self.vocab_size - 1)

        if not self.retain_dropout:
            self.model.eval()
        if isinstance(self.model.decoder, IncrementalDecoder):
            incremental_state = {}
        else:
            incremental_state = None

        # compute the encoder output for each beam
        encoder_out, encoder_padding_mask = self.model.encoder(
            src_tokens.repeat(1, beam_size).view(-1, srclen),
            src_lengths.expand(beam_size, src_lengths.numel()).t().contiguous().view(-1),
        )

        # initialize buffers
        scores = src_tokens.data.new(bsz * beam_size, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
        tokens_buf = tokens.clone()
        # FIXME: the first token should be bos?
        tokens[:, 0] = self.eos
        attn, attn_buf = None, None
        nonpad_idxs = None

        # list of completed sentences
        finalized = [[] for _ in range(bsz)]
        finished = [False for _ in range(bsz)]
        worst_finalized = [{"idx": None, "score": -math.inf} for _ in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == maxlen or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= maxlen**self.len_penalty
                if worst_finalized[sent]["score"] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.
            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.
            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1 : step + 2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos
            attn_clone = (
                attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2] if attn is not None else None
            )

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i][nonpad_idxs[sent]]
                        _, alignment = hypo_attn.max(dim=0)
                    else:
                        hypo_attn = None
                        alignment = None

                    return {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": alignment,
                        "positional_scores": pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]["score"]:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]["idx"]
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]["score"])
                    worst_finalized[sent] = {
                        "score": s["score"],
                        "idx": idx,
                    }

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        for step in range(maxlen + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                # for i, model in enumerate(self.models):
                if isinstance(self.model.decoder, IncrementalDecoder):
                    self.model.decoder.reorder_incremental_state(incremental_state, reorder_state)
                encoder_out, encoder_padding_mask = self.model.encoder.reorder_encoder_out(
                    encoder_out, encoder_padding_mask, reorder_state
                )

            probs, avg_attn_scores = self._decode(
                tokens[:, : step + 1], encoder_out, encoder_padding_mask, incremental_state
            )
            if step == 0:
                # at the first step all hypotheses are equally likely, so use
                # only the first beam
                probs = probs.unfold(0, 1, beam_size).squeeze(2).contiguous()
                scores = scores.type_as(probs)
                scores_buf = scores_buf.type_as(probs)
            elif not self.sampling:
                # make probs contain cumulative scores for each hypothesis
                probs.add_(scores[:, step - 1].view(-1, 1))

            probs[:, self.pad] = -math.inf  # never select pad
            probs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), maxlen + 2)
                    attn_buf = attn.clone()
                    nonpad_idxs = src_tokens.ne(self.pad)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            cand_beams = buffer("cand_beams")
            if step < maxlen:
                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    probs_slice = probs.view(bsz, -1, probs.size(-1))[:, 0, :]
                    cand_scores = torch.gather(
                        probs_slice, dim=1, index=prefix_tokens[:, step].view(-1, 1).data
                    ).expand(-1, cand_size)
                    cand_indices = prefix_tokens[:, step].view(-1, 1).expand(bsz, cand_size).data
                    cand_beams.resize_as_(cand_indices).fill_(0)
                elif self.sampling:
                    assert self.pad == 1, "sampling assumes the first two symbols can be ignored"

                    if self.sampling_topk > 0:
                        values, indices = probs[:, 2:].topk(self.sampling_topk)
                        exp_probs = values.div_(self.sampling_temperature).exp()
                        if step == 0:
                            cand_indices = torch.multinomial(exp_probs, beam_size, replacement=True)
                        else:
                            cand_indices = torch.multinomial(exp_probs, 1, replacement=True)
                        cand_scores = torch.gather(exp_probs, dim=1, index=cand_indices)
                        cand_indices = torch.gather(indices, dim=1, index=cand_indices)
                        cand_indices.add_(2)
                    else:
                        exp_probs = (
                            probs.div_(self.sampling_temperature).exp_().view(-1, self.vocab_size)
                        )

                        if step == 0:
                            # we exclude the first two vocab items, one of which is pad
                            cand_indices = torch.multinomial(
                                exp_probs[:, 2:], beam_size, replacement=True
                            )
                        else:
                            cand_indices = torch.multinomial(exp_probs[:, 2:], 1, replacement=True)

                        cand_indices.add_(2)
                        cand_scores = torch.gather(exp_probs, dim=1, index=cand_indices)

                    cand_scores.log_()
                    cand_indices = cand_indices.view(bsz, -1).repeat(1, 2)
                    cand_scores = cand_scores.view(bsz, -1).repeat(1, 2)
                    if step == 0:
                        cand_beams = torch.zeros(bsz, cand_size).type_as(cand_indices)
                    else:
                        cand_beams = torch.arange(0, beam_size).repeat(bsz, 2).type_as(cand_indices)
                        # make scores cumulative
                        cand_scores.add_(
                            torch.gather(
                                scores[:, step - 1].view(bsz, beam_size),
                                dim=1,
                                index=cand_beams,
                            )
                        )
                else:
                    # take the best 2 x beam_size predictions. We'll choose the first
                    # beam_size of these which don't predict eos to continue with.
                    cand_scores, cand_indices = torch.topk(
                        probs.view(bsz, -1),
                        k=min(
                            cand_size, probs.view(bsz, -1).size(1) - 1
                        ),  # -1 so we never select pad
                    )
                    cand_beams = torch.div(cand_indices, self.vocab_size, rounding_mode="trunc")
                    cand_indices.fmod_(self.vocab_size)
            else:
                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest prob of EOS right now
                eos_scores, eos_bbsz_idx = torch.sort(
                    probs[:, self.eos],
                    descending=True,
                )
                num_remaining_sent -= len(finalize_hypos(step, eos_bbsz_idx, eos_scores))
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)

            finalized_sents = set()
            if step >= self.minlen:
                # only consider eos when it's among the top beam_size indices
                eos_bbsz_idx = torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                )
                if eos_bbsz_idx.numel() > 0:
                    eos_scores = torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                    )
                    finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores, cand_scores)
                    num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < maxlen

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(bsz).type_as(cand_indices)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)

                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            _ignore, active_hypos = torch.topk(
                active_mask,
                k=beam_size,
                dim=1,
                largest=False,
            )
            active_bbsz_idx = torch.gather(
                cand_bbsz_idx,
                dim=1,
                index=active_hypos,
            )
            active_scores = torch.gather(
                cand_scores,
                dim=1,
                index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, : step + 1],
                dim=0,
                index=active_bbsz_idx,
                out=tokens_buf[:, : step + 1],
            )
            torch.gather(
                cand_indices,
                dim=1,
                index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step],
                    dim=0,
                    index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores,
                dim=1,
                index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, : step + 2],
                    dim=0,
                    index=active_bbsz_idx,
                    out=attn_buf[:, :, : step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r["score"], reverse=True)

        return finalized

    def _decode(self, tokens, encoder_out, encoder_padding_mask, incremental_state, log_probs=True):
        with torch.no_grad():
            if incremental_state is not None:
                decoder_out = list(
                    self.model.decoder(
                        tokens,
                        encoder_out,
                        encoder_padding_mask,
                        incremental_state=incremental_state,
                    )
                )
            else:
                decoder_out = list(self.model.decoder(tokens, encoder_out, encoder_padding_mask))
            decoder_out[0] = decoder_out[0][:, -1, :]
            attn = decoder_out[1]
            if isinstance(attn, torch.Tensor) and attn.numel() == 0:
                attn = None
            if attn is not None:
                attn = attn[:, -1, :]

        logits = decoder_out[0]
        if log_probs:
            probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        else:
            probs = F.softmax(logits, dim=-1, dtype=torch.float32)

        return probs, attn


if __name__ == "__main__":

    ...
