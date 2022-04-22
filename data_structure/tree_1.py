import os

class TreeNode():
    '''定义树节点'''
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

class BinTree():
    '''创建二叉树(完全二叉树)'''
    def __init__(self):
        self.root = None
        self.ls = []    # 定义列表用于存储节点地址

    def add(self, data):
        '''
        定义add方法,向树结构中添加元素,
        创建的过程也是一个按照 根, 左, 右 的顺序进行入栈操作
        '''
        node = TreeNode(data)
        if self.root == None:   # 如果根节点为None，添加根节点，并将地址添加进ls中
            self.root = node
            self.ls.append(node)
        else:
            rootNode = self.ls[0]   # 将第一个元素设为根节点
            if rootNode.left == None:
                rootNode.left = node
                self.ls.append(node)
            elif rootNode.right == None:
                rootNode.right = node
                self.ls.append(node)
                # 按照 根左右 的顺序来创建，只有当右节点也已有了元素，才能将 根 弹出
                self.ls.pop(0)  # 弹出 self.ls 第一个位置处的元素

    def preOrderTraversal(self, root):
        '''前序遍历(根左右)递归实现'''
        node = root
        if node == None:
            return 
        print(node.data)
        self.preOrderTraversal(node.left)
        self.preOrderTraversal(node.right)

    def inOrderTraversal(self, root):
        '''中序遍历'''
        node = root
        if root == None:
            return 
        self.inOrderTraversal(node.left)
        print(node.data)
        self.inOrderTraversal(node.right)

    def postOrderTraversal(self, root):
        '''后序遍历'''
        node = root
        if node == None:
            return
        self.postOrderTraversal(node.left)
        self.postOrderTraversal(node.right)
        print(node.data)

    def levelOrder(self, root):
        '''层序遍历'''
        node = root
        if node == None:
            return
        queue = []  # 创建队列
        result = []
        queue.append(node)  # 根节点入队
        while queue:
            node = queue.pop(0)
            result.append(node.data)
            if node.left != None:
                queue.append(node.left)
            if node.right != None:
                queue.append(node.right)
        print(result)
        return result

    
if __name__ == "__main__":
    tree = BinTree()
    for i in range(1, 11):
        tree.add(i)

    # tree.preOrderTraversal(tree.root)
    # tree.inOrderTraversal(tree.root)
    # tree.postOrderTraversal(tree.root)
    tree.levelOrder(tree.root)