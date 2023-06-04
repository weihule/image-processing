def getMaxRepeatSubstringLength(inputStr):
    # length = len(inputStr)
    # print([p for p in reversed(range(length//2))])
    # for i in reversed(range(length//2)):
    #     print(i)
    #     count = 0
    #     for j in range(length - i):
    #         if inputStr[j] == inputStr[j+i]:
    #             count = count + 1
    #         else:
    #             if length - j <= 2 * i:
    #                 break
    #             count = 0
    #         if count == i:
    #             return count * 2
    # return 0

    max_len = i = 0
    j =1
    # i 是子串的起始位置, j是子串的末尾 故 j-i 就是子串长度
    while j<len(inputStr):
        if inputStr[i:j] in inputStr[j:]:       # 查看子串是否连续出现两次
            if inputStr[i:j]==inputStr[j:j+j-i]:    # 查看两个子串是否是连续的
                max_len = max(max_len, j-i)
            j +=1
            i -=1
        i +=1
    return 2*max_len


def decode_arr(arr):
    maps = {'A': 10, 'B': 11, 'C': 12,
            'D': 13, 'E': 14, 'F': 15}
    temp = []
    for idx in range(len(arr)-1, -1, -1):
        if arr[idx] != '%':
            temp.append(arr[idx])
        else:
            # temp弹出
            new_v = 0
            for i in range(1, -1, -1):
                cur_v = temp.pop()
                if cur_v in maps:
                    cur_v = maps[cur_v]

                new_v += int(cur_v) * (16 ** i)
            new_v = chr(new_v)
            temp.append(new_v)
    print(temp)
    temp = temp[::-1]
    
    res = ''.join(temp)
    
    return res


if __name__ == "__main__":
    # inputStr = 'ababc'       # 'ababc'   'xabcabcx'
    # res = getMaxRepeatSubstringLength(inputStr)
    # print(res)

    arr = "%%32F"
    res = decode_arr(arr)