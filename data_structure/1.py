
def cal_res(res):
    if len(set(res)) == len(res):
        return len(res) 

def main():
    nums = int(input().strip())

    arrs = []
    for _ in range(nums):
        arrs.append(str(input().strip()))
    
    for idx, arr in enumerate(arrs):
        print('Case #{}: {}'.formate(idx, res))
