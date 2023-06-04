
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def hasCycle(head) -> bool:
    # if (head is None or head.next is None):
    #     return False

    # if head.next == head:
    #     return True
    # nextNode = head.next

    # head.next = head

    # return hasCycle(nextNode)
    if head is None or head.next is None:
        return False
    
    if head == head.next:
        return True
    
    next_node = head.next

    # 断掉当前节点
    head.next = head

    return hasCycle(next_node)


def traverse(head):
    while head:
        value = head.val
        print(id(head), value)
        head = head.next


if __name__ == "__main__":
    node1 = ListNode(10)
    node2 = ListNode(8)
    node3 = ListNode(12)
    node1.next = node2
    node2.next = node3
    # node3.next = node2

    # traverse(node1)
    # print('='*20)

    # res = hasCycle(node1)
    # print(res)

    nums1 = [1, 2, 3, 0, 0, 0]
    nums = [5, 6, 7]
    print(nums1, id(nums1))
    nums1 = nums
    print(nums1, id(nums1))