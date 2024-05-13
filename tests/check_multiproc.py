from tqdm import tqdm
from multiprocessing import Pool


def f(x, y):
    return x**2 + y**2


if __name__ == "__main__":
    array = list(zip(range(50), range(50, 100)))

    pbar = tqdm(total=len(array))
    with Pool(processes=10) as pool:
        res = [pool.apply_async(f, args=(x, y), callback=pbar.update()) for x, y in array]
        pool.close()
        pool.join()

    res_value = tuple(r.get() for r in res)
    print(res_value)
