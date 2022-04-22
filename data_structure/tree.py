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
		self.ls = []		# 定义列表用于存储节点地址

	def add(self, data):
		'''定义add方法, 向树结构中添加元素'''
		node = TreeNode(data)
		# print(node)
		if self.root == None:		# 若根节点为None，添加根节点，并把根节点的地址值添加到 self.ls 中
			self.root = node
			self.ls.append(self.root)
		else:
			rootNode = self.ls[0]	# 将第一个元素设为根节点
			if rootNode.left == None:
				rootNode.left = node
				self.ls.append(rootNode.left)
			elif rootNode.right == None:
				rootNode.right = node
				self.ls.append(rootNode.right)
				self.ls.pop(0)		# 弹出 self.ls 第一个位置处的元素

	def preOrederTraversal(self, root):
		'''前序遍历(根左右)：递归实现'''
		if root == None:
			return 
		print(root.data)		# 打印根节点的数据项
		self.preOrederTraversal(root.left)
		self.preOrederTraversal(root.right)

	def preOrderStack1(self, root):
		'''前序遍历(根左右): 堆栈实现1'''
		if root == None:
			return
		stack = []
		result = []
		node = root
		while node or stack:		# 当node不为None或stack不为空时进入循环
			print(result, stack)
			while node:		# 寻找当前节点的左子节点，并将其地址添加到stack中
				result.append(node.data)	# 将当前节点的数据项添加到result中
				stack.append(node)
				node = node.left        # 当某节点不再有子节点时，推出内循环
			node = stack.pop()		# 将当前节点pop出stack，获取其地址值
			node = node.right       # 寻找当前节点的右子节点
		print(result)

	def preOrderStack2(self, root):
		'''前序遍历(根左右): 堆栈实现2'''
		stack = []
		result = []
		node = root
		if node:
			stack.append(node)
		while stack:
			node = stack.pop()
			result.append(node.data)	
			if node.right:
				stack.append(node.right)	# 右
			if node.left:
				stack.append(node.left)		# 左
		print(result)

	def inOrderTraversal(self, root):
		'''中序遍历(左根右): 递归实现'''
		node = root
		if node == None:
			return
		self.inOrderTraversal(node.left)
		print(node.data)
		self.inOrderTraversal(node.right)

	def inOrderStack(self, root):
		'''中序遍历(左根右): 堆栈实现'''
		if root == None:
			return
		stack = []
		result = []
		node = root
		while node or stack:
			while node:
				stack.append(node)
				node = node.left	# 寻找当前节点的左子节点，直到当前节点无左子节点跳出内循环
			node = stack.pop()		# 将当前节点pop出stack，获取当前节点的地址值
			result.append(node.data)
			node = node.right		# 需要当前节点的右子节点
		print(result)

if __name__ == "__main__":
	tree = BinTree()
	for i in range(1, 11):
		tree.add(i)
	# tree.preOrederTraversal(tree.root)
	# tree.preOrderStack2(tree.root)
	# tree.inOrderStack1(tree.root)
	tree.inOrderStack(tree.root)