
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
		if self.root is None:		# 若根节点为None，添加根节点，并把根节点的地址值添加到 self.ls 中
			self.root = node
			self.ls.append(self.root)
		else:
			rootNode = self.ls[0]	# 将第一个元素设为根节点
			if rootNode.left is None:
				rootNode.left = node
				self.ls.append(rootNode.left)
			elif rootNode.right is None:
				rootNode.right = node
				self.ls.append(rootNode.right)
				self.ls.pop(0)		# 当前的右子树填充完之后, 弹出 self.ls 第一个位置处的元素

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
				node = node.left        # 当某节点不再有子节点时，退出内循环
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


# 20220807
class TreeNode1():
	def __init__(self, val, left=None, right=None):
		self.val = val
		self.left = left
		self.right = right


class BinTree1():

	# 初始化完全二叉树
	def __init__(self):
		self.root = None
		self.ls = []
	
	def add_val(self, value):
		node = TreeNode1(value)
		if self.root is None:
			self.root = node
			self.ls.append(node)
		else:
			# cur_node = self.ls.pop(0)	# 这里如果弹出, 就无法添加右节点, 只有在右节点添加之后才能弹出
			cur_node = self.ls[0]
			if cur_node.left is None:
				cur_node.left = node
				self.ls.append(node)
			elif cur_node.right is None:
				cur_node.right = node
				self.ls.append(node)
				self.ls.pop(0)
	
	def pre_traverse(self, node):
		if node is None:
			return 
		print(node.val)
		self.pre_traverse(node.left)
		self.pre_traverse(node.right)

	def inorder_traverse(self, node):
		if node is None:
			return 
		self.inorder_traverse(node.left)
		print(node.val)
		self.inorder_traverse(node.right)

	def postorder_traverse(self, node):
		if node is None:
			return 
		self.inorder_traverse(node.left)
		self.inorder_traverse(node.right)
		print(node.val)

	def bfs(self, node):
		"""
		这个是层序遍历并将所有的值存放在一个列表中
		"""
		queue = []
		queue.append(node)

		while len(queue) != 0:
			cur_node = queue.pop(0)
			print(cur_node.val)
			if cur_node.left:
				queue.append(cur_node.left)
			if cur_node.right:
				queue.append(cur_node.right)

	def bfs_layer(self, node):
		"""
		将每层结果单独放在一个子列表中
		"""
		if node is None:
			return []
		res = []
		queue = []
		queue.append(node)

		while queue:
			per_layer_len = len(queue)	# 每层的节点数
			sub_list = []
			for _ in range(per_layer_len):
				cur_node = queue.pop(0)
				sub_list.append(cur_node.val)
				if cur_node.left:
					queue.append(cur_node.left)		# 入的是 queue 这个栈
				if cur_node.right:
					queue.append(cur_node.right)
			res.append(sub_list)
		return res


	def max_depth(self, node):
		if node is None:
			return 0
		left_depth = self.max_depth(node.left)
		right_depth = self.max_depth(node.right)

		# 当前节点的最大深度等于，左右子节点的最大深度 + 1
		return max(left_depth, right_depth) + 1


class BinTree2():
	# 初始化二叉树
	def __init__(self):
		self.root = None
		self.ls = []

	def add_val(self, value):
		node = TreeNode1(value)
		if self.root is None:
			self.root = node
			self.ls.append(node)
		else:
			cur_root_node = self.ls[0]
			if cur_root_node.left is None:
				cur_root_node.left = node
				self.ls.append(node)
			elif cur_root_node.right is None:
				cur_root_node.right = node
				self.ls.append(node)
				self.ls.pop(0)
	
	def pre_traverse(self, node):
		if node is None:
			return 
		print(node.val)
		self.pre_traverse(node.left)
		self.pre_traverse(node.right)

	def bfs(self, node):
		queue = []
		queue.append(node)

		while len(queue) != 0:
			cur_node = queue.pop(0)
			print(cur_node.val)
			if cur_node.left:
				queue.append(cur_node.left)
			if cur_node.right:
				queue.append(cur_node.right)

if __name__ == "__main__":
	tree = BinTree()
	for i in range(1, 11):
		tree.add(i)
	# tree.preOrederTraversal(tree.root)
	# tree.preOrderStack2(tree.root)
	# tree.inOrderStack1(tree.root)
	# tree.inOrderStack(tree.root)

	print('='*20)

	tree1 = BinTree1()
	for i in range(1, 11):
		tree1.add_val(i)
	# tree1.pre_traverse(tree1.root)
	# tree1.inorder_traverse(tree1.root)
	print('---')
	# tree1.postorder_traverse(tree1.root)
	tree1.bfs(tree1.root)
	res = tree1.bfs_layer(tree1.root)
	print(res)
	# tree_depth = tree1.max_depth(tree1.root)
	# print(tree_depth)

	tree2 = BinTree2()
	for i in range(1, 11):
		tree2.add_val(i)
	# tree2.pre_traverse(tree2.root)
	# tree2.bfs(tree2.root)