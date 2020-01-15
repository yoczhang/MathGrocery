[toc]

---

##### main function:

```python
#!/usr/bin/env python3
# 
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# 导入 Poisson 有限元模型
from fealpy.fem.PoissonFEMModel import PoissonFEMModel

from fealpy.tools.show import showmultirate, show_error_table

# 问题维数
# d = int(sys.argv[1])
d = 2

if d == 1:
    from fealpy.pde.poisson_1d import CosData as PDE
elif d == 2:
    from fealpy.pde.poisson_2d import CosCosData as PDE
elif d == 3:
    from fealpy.pde.poisson_3d import CosCosCosData as PDE

# p = int(sys.argv[2]) # 有限元空间的次数
# n = int(sys.argv[3]) # 初始网格的加密次数
# maxit = int(sys.argv[4]) # 迭代加密的次数

p = 1  # 有限元空间的次数
n = 2  # 初始网格的加密次数
maxit = 3  # 迭代加密的次数

pde = PDE()  # 创建 pde 模型

# 误差类型与误差存储数组
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

# 自由度数组
Ndof = np.zeros(maxit, dtype=np.int)

# 创建初始网格对象 
mesh = pde.init_mesh(n)

for i in range(maxit):
    fem = PoissonFEMModel(pde, mesh, p, q=p + 2)  # 创建 Poisson 有限元模型
    ls = fem.solve()  # 求解
    Ndof[i] = fem.space.number_of_global_dofs()  # 获得空间自由度个数
    errorMatrix[0, i] = fem.L2_error()  # 计算 L2 误差
    errorMatrix[1, i] = fem.H1_semi_error()  # 计算 H1 误差
    if i < maxit - 1:
        mesh.uniform_refine()  # 一致加密网格

# 显示误差
show_error_table(Ndof, errorType, errorMatrix)
# 可视化误差收敛阶
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
```

##### CosCosData:

```python
class CosCosData:
    """
    -\Delta u = f
    u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=4, meshtype='tri', h=0.1):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        if meshtype is 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        if meshtype is 'quad':
            node = np.array([
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
                (0.5, 0),
                (1, 0.4),
                (0.3, 1),
                (0, 0.6),
                (0.5, 0.45)], dtype=np.float)
            cell = np.array([
                (0, 4, 8, 7), (4, 1, 5, 8),
                (7, 8, 6, 3), (8, 5, 2, 6)], dtype=np.int)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'stri':
            mesh = StructureQuadMesh([0, 1, 0, 1], h)
            return mesh
        else:
            raise ValueError("".format)


    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val


    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.cos(pi*x)*np.cos(pi*y)
        return val


    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val

    def dirichlet(self, p):
        return self.solution(p)

    def neuman(self, p, n):
        """ Neuman  boundary condition
        p: (NQ, NE, 2)
        n: (NE, 2)
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        return val

    def robin(self, p):
        pass
```



##### TriangleMesh:

```python
class TriangleMesh(Mesh2d):
    def __init__(self, node, cell):

        self.node = node
        N = node.shape[0]
        self.ds = TriangleMeshDataStructure(N, cell)
        if node.shape[1] == 2:
            self.meshtype = 'tri'
        elif node.shape[1] == 3:
            self.meshtype = 'stri'

        self.itype = cell.dtype
        self.ftype = node.dtype

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.meshdata = {}

    def vtk_cell_type(self):
        VTK_TRIANGLE = 5
        return VTK_TRIANGLE

    def integrator(self, k, etype='cell'):
        if etype in ['cell', 2]:
            return TriangleQuadrature(k)
        elif etype in ['edge', 'face', 1]:
            return GaussLegendreQuadrature(k)

    def copy(self):
        return TriangleMesh(self.node.copy(), self.ds.cell.copy());

    def delete_cell(self, threshold):
        NN = self.number_of_nodes()

        cell = self.entity('cell')
        bc = mesh.entity_barycenter('cell')
        isKeepCell = ~threshold(bc)
        cell = cell[isKeepCell]

        isValidNode = np.zeros(NN, dtype=np.bool)
        isValidNode[cell] = True
        node = node[isValidNode]

        idxMap = np.zeros(NN, dtype=mesh.itype)
        idxMap[isValidNode] = range(isValidNode.sum())
        cell = idxMap[cell]
        self.node = node
        NN = len(node)
        self.ds.reinit(NN, cell)

    def to_quadmesh(self):
        from ..mesh import QuadrangleMesh

        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        node0 = self.entity('node')
        cell0 = self.entity('cell')
        ec = self.entity_barycenter('edge')
        cc = self.entity_barycenter('cell')
        cell2edge = self.ds.cell_to_edge()

        node = np.r_['0', node0, ec, cc]
        cell = np.zeros((3*NC, 4), dtype=self.itype)
        idx = np.arange(NC)
        cell[:NC, 0] = NN + NE + idx
        cell[:NC, 1] = cell2edge[:, 0] + NN
        cell[:NC, 2] = cell0[:, 2]
        cell[:NC, 3] = cell2edge[:, 1] + NN

        cell[NC:2*NC, 0] = cell[:NC, 0]
        cell[NC:2*NC, 1] = cell2edge[:, 1] + NN
        cell[NC:2*NC, 2] = cell0[:, 0]
        cell[NC:2*NC, 3] = cell2edge[:, 2] + NN

        cell[2*NC:3*NC, 0] = cell[:NC, 0]
        cell[2*NC:3*NC, 1] = cell2edge[:, 2] + NN
        cell[2*NC:3*NC, 2] = cell0[:, 1]
        cell[2*NC:3*NC, 3] = cell2edge[:, 0] + NN
        return QuadrangleMesh(node, cell)


    def line_walk(self, p):
        NC = self.number_of_cells()
        NP = p.shape[0]
        cell = self.entity('cell')

        cidx = np.random.randint(0, NC, NP)
        isNotOK = np.ones(NP, dtype=np.bool)
        while isNotOK.sum() > 0:
            idx0, = np.find(isNotOK)
            cidx0 = cidx[isNotOK]
            pp = p[isNotOK]

            a = np.zeros((len(cidx0), 3), dtype=self.ftype)
            p0 = cell[cidx0, 0]
            p1 = cell[cidx0, 1]
            p2 = cell[cidx0, 2]

            v0 = p0 - pp
            v1 = p1 - pp
            v2 = p2 - pp
            a[:, 0] = np.cross(v1, v2)
            a[:, 1] = np.cross(v2, v0)
            a[:, 2] = np.cross(v0, v1)
            idx = np.argmin(a, axis=-1)

            isOK = a[range(a.shape[0]), idx] >= 0

    def circumcenter(self):
        node = self.node
        cell = self.ds.cell
        dim = self.geo_dimension()

        v0 = node[cell[:,2],:] - node[cell[:,1],:]
        v1 = node[cell[:,0],:] - node[cell[:,2],:]
        v2 = node[cell[:,1],:] - node[cell[:,0],:]
        nv = np.cross(v2, -v1)
        if dim == 2:
            area = nv/2.0
            x2 = np.sum(node**2, axis=1, keepdims=True)
            w0 = x2[cell[:,2]] + x2[cell[:,1]]
            w1 = x2[cell[:,0]] + x2[cell[:,2]]
            w2 = x2[cell[:,1]] + x2[cell[:,0]]
            W = np.array([[0, -1],[1, 0]], dtype=self.ftype)
            fe0 = w0*v0@W
            fe1 = w1*v1@W
            fe2 = w2*v2@W
            c = 0.25*(fe0 + fe1 + fe2)/area.reshape(-1,1)
            R = np.sqrt(np.sum((c-node[cell[:,0], :])**2,axis=1))
        elif dim == 3:
            length = np.sqrt(np.sum(nv**2, axis=1))
            n = nv/length.reshape((-1, 1))
            l02 = np.sum(v1**2, axis=1, keepdims=True)
            l01 = np.sum(v2**2, axis=1, keepdims=True)
            d = 0.5*(l02*np.cross(n, v2) + l01*np.cross(-v1, n))/length.reshape(-1, 1)
            c = node[cell[:, 0]] + d
            R = np.sqrt(np.sum(d**2, axis=1))
        return c, R

    def angle(self):
        NC = self.number_of_cells()
        cell = self.ds.cell
        node = self.node
        localEdge = self.ds.local_edge()
        angle = np.zeros((NC, 3), dtype=self.ftype)
        for i,(j,k) in zip(range(3),localEdge):
            v0 = node[cell[:,j]] - node[cell[:,i]]
            v1 = node[cell[:,k]] - node[cell[:,i]]
            angle[:,i] = np.arccos(np.sum(v0*v1, axis=1)/np.sqrt(np.sum(v0**2, axis=1) * np.sum(v1**2, axis=1)))
        return angle

    def edge_swap(self):
        while True:
            # Construct necessary data structure
            edge2cell = self.ds.edge_to_cell()
            cell2edge = self.ds.cell_to_edge()

            # Find non-Delaunay edges
            angle = self.angle()
            asum = np.sum(angle[edge2cell[:, 0:2], edge2cell[:, 2:4]], axis=1)
            isNonDelaunayEdge = (asum > np.pi) & (edge2cell[:,0] != edge2cell[:,1])

            return isNonDelaunayEdge
            
            if np.sum(isNonDelaunayEdge) == 0:
                break
            # Find dependent set of swap edges
            isCheckCell = np.sum(isNonDelaunayEdge[cell2edge], axis=1) > 1
            if np.any(isCheckCell):
                ac = asum[cell2edge[isCheckCell, :]]
                isNonDelaunayEdge[cell2edge[isCheckCell, :]] = False
                I = np.argmax(ac, axis=1)
                isNonDelaunayEdge[cell2edge[isCheckCell, I]] = True

            if np.any(isNonDelaunayEdge):
                cell = self.ds.cell
                pnext = np.array([1, 2, 0])
                idx = edge2cell[isNonDelaunayEdge, 2]
                p0 = cell[edge2cell[isNonDelaunayEdge, 0], idx]
                p1 = cell[edge2cell[isNonDelaunayEdge, 0], pnext[idx]] 
                idx = edge2cell[isNonDelaunayEdge, 3]
                p2 = cell[edge2cell[isNonDelaunayEdge, 1], idx]
                p3 = cell[edge2cell[isNonDelaunayEdge, 1], pnext[idx]]
                cell[edge2cell[isNonDelaunayEdge, 0], 0] = p1
                cell[edge2cell[isNonDelaunayEdge, 0], 1] = p2
                cell[edge2cell[isNonDelaunayEdge, 0], 2] = p0

                cell[edge2cell[isNonDelaunayEdge, 1], 0] = p3
                cell[edge2cell[isNonDelaunayEdge, 1], 1] = p0
                cell[edge2cell[isNonDelaunayEdge, 1], 2] = p2

                NN = self.number_of_nodes()
                self.ds.reinit(NN, cell)

    def uniform_refine(self, n=1, surface=None, returnim=False):
        if returnim:
            nodeIMatrix = []
            cellIMatrix = []
        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            NE = self.number_of_edges()
            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')
            cell2edge = self.ds.cell_to_edge()
            edge2newNode = np.arange(NN, NN+NE)
            newNode = (node[edge[:,0],:] + node[edge[:,1],:])/2.0

            if returnim:
                A = coo_matrix((np.ones(NN), (range(NN), range(NN))), shape=(NN+NE, NN), dtype=self.ftype)
                A += coo_matrix((0.5*np.ones(NE), (range(NN, NN+NE), edge[:, 0])), shape=(NN+NE, NN), dtype=self.ftype)
                A += coo_matrix((0.5*np.ones(NE), (range(NN, NN+NE), edge[:, 1])), shape=(NN+NE, NN), dtype=self.ftype)
                nodeIMatrix.append(A.tocsr())
                B = eye(NC, dtype=self.ftype)
                B = bmat([[B], [B], [B], [B]])
                cellIMatrix.append(B.tocsr())

            if surface is not None:
                newNode, _ = surface.project(newNode)

            self.node = np.concatenate((node, newNode), axis=0)
            p = np.r_['-1', cell, edge2newNode[cell2edge]]
            cell = np.r_['0', p[:, [0, 5, 4]], p[:, [5, 1, 3]], p[:, [4, 3, 2]], p[:, [3, 4, 5]]]
            NN = self.node.shape[0]
            self.ds.reinit(NN, cell)
        if returnim:
            return nodeIMatrix, cellIMatrix

    def uniform_bisect(self, n=1):
        for i in range(n):
            self.bisect()

    def bisect(self, isMarkedCell='all', returnim=False, refine=None):

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NE = self.number_of_edges()

        if isMarkedCell is 'all':
            isMarkedCell = np.ones(NC, dtype=np.bool)

        cell = self.entity('cell')
        edge = self.entity('edge')

        cell2edge = self.ds.cell_to_edge()
        cell2cell = self.ds.cell_to_cell()

        isCutEdge = np.zeros((NE,), dtype=np.bool)

        markedCell, = np.nonzero(isMarkedCell)
        while len(markedCell)>0:
            isCutEdge[cell2edge[markedCell, 0]]=True
            refineNeighbor = cell2cell[markedCell, 0]
            markedCell = refineNeighbor[~isCutEdge[cell2edge[refineNeighbor,0]]]

        edge2newNode = np.zeros((NE,), dtype=self.itype)
        edge2newNode[isCutEdge] = np.arange(NN, NN+isCutEdge.sum())

        node = self.node
        newNode =0.5*(node[edge[isCutEdge,0],:] + node[edge[isCutEdge,1],:])
        self.node = np.concatenate((node, newNode), axis=0)
        cell2edge0 = cell2edge[:, 0]

        if returnim:
            nn = len(newNode)
            IM = coo_matrix((np.ones(NN), (np.arange(NN), np.arange(NN))),
                    shape=(NN+nn, NN), dtype=self.ftype)
            val = np.full(nn, 0.5)
            IM += coo_matrix(
                    (
                        val,
                        (
                            NN+np.arange(nn),
                            edge[isCutEdge, 0]
                        )
                    ), shape=(NN+nn, NN), dtype=self.ftype)
            IM += coo_matrix(
                    (
                        val,
                        (
                            NN+np.arange(nn),
                            edge[isCutEdge, 1]
                        )
                    ), shape=(NN+nn, NN), dtype=self.ftype)

        for k in range(2):
            idx, = np.nonzero(edge2newNode[cell2edge0]>0)
            nc = len(idx)
            if nc == 0:
                break
            L = idx
            R = np.arange(NC, NC+nc)
            p0 = cell[idx,0]
            p1 = cell[idx,1]
            p2 = cell[idx,2]
            p3 = edge2newNode[cell2edge0[idx]]
            cell = np.concatenate((cell, np.zeros((nc,3), dtype=self.itype)), axis=0)
            cell[L,0] = p3
            cell[L,1] = p0
            cell[L,2] = p1
            cell[R,0] = p3
            cell[R,1] = p2
            cell[R,2] = p0
            if k == 0:
                cell2edge0 = np.zeros((NC+nc,), dtype=self.itype)
                cell2edge0[0:NC] = cell2edge[:,0]
                cell2edge0[L] = cell2edge[idx,2]
                cell2edge0[R] = cell2edge[idx,1]
            NC = NC+nc

        NN = self.node.shape[0]
        self.ds.reinit(NN, cell)

        if returnim:
            return IM.tocsr()

    def label(self, node=None, cell=None, cellidx=None):
        """单元顶点的重新排列，使得cell[:, [1, 2]] 存储了单元的最长边
        Parameter
        ---------

        Return
        ------
        cell ： in-place modify

        """

        rflag = False
        if node is None:
            node = self.entity('node')

        if cell is None:
            cell = self.entity('cell')
            rflag = True

        if cellidx is None:
            cellidx = np.arange(len(cell))

        NC = cellidx.shape[0]
        localEdge = self.ds.localEdge
        totalEdge = cell[cellidx][:, localEdge].reshape(
                -1, localEdge.shape[1])
        NE = totalEdge.shape[0]
        length = np.sum(
                (node[totalEdge[:, 1]] - node[totalEdge[:, 0]])**2,
                axis = -1)
        length += 0.1*np.random.rand(NE)*length
        cellEdgeLength = length.reshape(NC, 3)
        lidx = np.argmax(cellEdgeLength, axis=-1)

        flag = (lidx == 1)
        if  sum(flag) > 0:
            cell[cellidx[flag], :] = cell[cellidx[flag]][:, [1, 2, 0]]

        flag = (lidx == 2)
        if sum(flag) > 0:
            cell[cellidx[flag], :] = cell[cellidx[flag]][:, [2, 0, 1]]

        if rflag == True:
            self.ds.construct()

    def adaptive_options(
            self,
            method='mean',
            maxrefine=5,
            maxcoarsen=0,
            theta=1.0,
            HB=None,
            imatrix=False,
            data=None,
            disp=True,
            ):

        options = {
                'method': method,
                'maxrefine': maxrefine,
                'maxcoarsen': maxcoarsen,
                'theta': theta,
                'data': data,
                'HB': HB,
                'imatrix': imatrix,
                'disp': disp
            }
        return options


    def adaptive(self, eta, options):
        theta = options['theta']
        if options['method'] is 'mean':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.mean(eta)))
                )
        elif options['method'] is 'max':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.max(eta)))
                )
        elif options['method'] is 'median':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.median(eta)))
                )
        elif options['method'] is 'min':
            options['numrefine'] = np.around(
                    np.log2(eta/(theta*np.min(eta)))
                )
        else:
            raise ValueError(
                    "I don't know anyting about method %s!".format(options['method']))

        flag = options['numrefine'] > options['maxrefine']
        options['numrefine'][flag] = options['maxrefine']
        flag = options['numrefine'] < -options['maxcoarsen']
        options['numrefine'][flag] = -options['maxcoarsen']

        # refine
        NC = self.number_of_cells()
        print("Number of cells before:", NC)
        isMarkedCell = (options['numrefine'] > 0)
        while sum(isMarkedCell) > 0:
            self.bisect_1(isMarkedCell, options)
            print("Number of cells after refine:", self.number_of_cells())
            isMarkedCell = (options['numrefine'] > 0)

        # coarsen
        if options['maxcoarsen'] > 0:
            isMarkedCell = (options['numrefine'] < 0)
            while sum(isMarkedCell) > 0:
                NN0 = self.number_of_cells()
                self.coarsen_1(isMarkedCell, options)
                NN = self.number_of_cells()
                if NN == NN0:
                    break
                print("Number of cells after coarsen:", self.number_of_cells())
                isMarkedCell = (options['numrefine'] < 0)

    def bisect_1(self, isMarkedCell=None, options={'disp': True}):

        GD = self.geo_dimension()
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NN0 = NN  # 记录下二分加密之前的节点数目

        if isMarkedCell is None:
            # 默认加密所有的单元
            markedCell = np.arange(NC, dtype=self.itype)
        else:
            markedCell, = np.nonzero(isMarkedCell)

        # allocate new memory for node and cell
        node = np.zeros((5*NN, GD), dtype=self.ftype)
        cell = np.zeros((2*NC, 3), dtype=self.itype)

        if ('numrefine' in options) and (options['numrefine'] is not None):
            options['numrefine'] = np.r_[options['numrefine'], np.zeros(NC)]

        node[:NN] = self.entity('node')
        cell[:NC] = self.entity('cell')

        # 用于存储网格节点的代数，初始所有节点都为第 0 代
        generation = np.zeros(NN + 2*NC, dtype=np.uint8)

        # 用于记录被二分的边及其中点编号
        cutEdge = np.zeros((4*NN, 3), dtype=self.itype)

        # 当前的二分边的数目
        nCut = 0
        # 非协调边的标记数组 
        nonConforming = np.ones(4*NN, dtype=np.bool)
        while len(markedCell) != 0:
            # 标记最长边
            self.label(node, cell, markedCell)

            # 获取标记单元的四个顶点编号
            p0 = cell[markedCell, 0]
            p1 = cell[markedCell, 1]
            p2 = cell[markedCell, 2]

            # 找到新的二分边和新的中点 
            nMarked = len(markedCell)
            p3 = np.zeros(nMarked, dtype=self.itype)

            if nCut == 0: # 如果是第一次循环 
                idx = np.arange(nMarked) # cells introduce new cut edges
            else:
                # all non-conforming edges
                ncEdge = np.nonzero(nonConforming[:nCut])
                NE = len(ncEdge)
                I = cutEdge[ncEdge][:, [2, 2]].reshape(-1)
                J = cutEdge[ncEdge][:, [0, 1]].reshape(-1)
                val = np.ones(len(I), dtype=np.bool)
                nv2v = csr_matrix(
                        (val, (I, J)),
                        shape=(NN, NN))
                i, j =  np.nonzero(nv2v[:, p1].multiply(nv2v[:, p2]))
                p3[j] = i
                idx, = np.nonzero(p3 == 0)

            if len(idx) != 0:
                # 把需要二分的边唯一化 
                NE = len(idx)
                cellCutEdge = np.array([p1[idx], p2[idx]])
                cellCutEdge.sort(axis=0)
                s = csr_matrix(
                    (
                        np.ones(NE, dtype=np.bool),
                        (
                            cellCutEdge[0, :],
                            cellCutEdge[1, :]
                        )
                    ), shape=(NN, NN))
                # 获得唯一的边 
                i, j = s.nonzero()
                nNew = len(i)
                newCutEdge = np.arange(nCut, nCut+nNew)
                cutEdge[newCutEdge, 0] = i
                cutEdge[newCutEdge, 1] = j
                cutEdge[newCutEdge, 2] = range(NN, NN+nNew)
                node[NN:NN+nNew, :] = (node[i, :] + node[j, :])/2.0
                nCut += nNew
                NN += nNew

                # 新点和旧点的邻接矩阵 
                I = cutEdge[newCutEdge][:, [2, 2]].reshape(-1)
                J = cutEdge[newCutEdge][:, [0, 1]].reshape(-1)
                val = np.ones(len(I), dtype=np.bool)
                nv2v = csr_matrix(
                        (val, (I, J)),
                        shape=(NN, NN))
                i, j =  np.nonzero(nv2v[:, p1].multiply(nv2v[:, p2]))
                p3[j] = i

            # 如果新点的代数仍然为 0
            idx = (generation[p3] == 0)
            cellGeneration = np.max(
                    generation[cell[markedCell[idx]]],
                    axis=-1)
            # 第几代点 
            generation[p3[idx]] = cellGeneration + 1
            cell[markedCell, 0] = p3
            cell[markedCell, 1] = p0
            cell[markedCell, 2] = p1
            cell[NC:NC+nMarked, 0] = p3
            cell[NC:NC+nMarked, 1] = p2
            cell[NC:NC+nMarked, 2] = p0

            if ('numrefine' in options) and (options['numrefine'] is not None):
                options['numrefine'][markedCell] -= 1
                options['numrefine'][NC:NC+nMarked] = options['numrefine'][markedCell]

            NC = NC + nMarked
            del cellGeneration, p0, p1, p2, p3

            # 找到非协调的单元 
            checkEdge, = np.nonzero(nonConforming[:nCut])
            isCheckNode = np.zeros(NN, dtype=np.bool)
            isCheckNode[cutEdge[checkEdge]] = True
            isCheckCell = np.sum(
                    isCheckNode[cell[:NC]],
                    axis= -1) > 0
            # 找到所有包含检查节点的单元编号 
            checkCell, = np.nonzero(isCheckCell)
            I = np.repeat(checkCell, 3)
            J = cell[checkCell].reshape(-1)
            val = np.ones(len(I), dtype=np.bool)
            cell2node = csr_matrix((val, (I, J)), shape=(NC, NN))
            i, j = np.nonzero(
                    cell2node[:, cutEdge[checkEdge, 0]].multiply(
                        cell2node[:, cutEdge[checkEdge, 1]]
                        ))
            markedCell = np.unique(i)
            nonConforming[checkEdge] = False
            nonConforming[checkEdge[j]] = True;

        if ('imatrix' in options) and (options['imatrix'] is True):
            nn = NN - NN0
            IM = coo_matrix(
                    (
                        np.ones(NN0),
                        (
                            np.arange(NN0),
                            np.arange(NN0)
                        )
                    ), shape=(NN, NN), dtype=self.ftype)
            cutEdge = cutEdge[:nn]
            val = np.full((nn, 2), 0.5, dtype=self.ftype)

            g = 2
            markedNode, = np.nonzero(generation == g)

            N = len(markedNode)
            while N != 0:
                nidx = markedNode - NN0
                i = cutEdge[nidx, 0]
                j = cutEdge[nidx, 1]
                ic = np.zeros((N, 2), dtype=self.ftype)
                jc = np.zeros((N, 2), dtype=self.ftype)
                ic[i < NN0, 0] = 1.0
                jc[j < NN0, 1] = 1.0
                ic[i >= NN0, :] = val[i[i >= NN0] - NN0, :]
                jc[j >= NN0, :] = val[j[j >= NN0] - NN0, :]
                val[markedNode - NN0, :] = 0.5*(ic + jc)
                cutEdge[nidx[i >= NN0], 0] = cutEdge[i[i >= NN0] - NN0, 0]
                cutEdge[nidx[j >= NN0], 1] = cutEdge[j[j >= NN0] - NN0, 1]
                g += 1
                markedNode, = np.nonzero(generation == g)
                N = len(markedNode)

            IM += coo_matrix(
                    (
                        val.flat,
                        (
                            cutEdge[:, [2, 2]].flat,
                            cutEdge[:, [0, 1]].flat
                        )
                    ), shape=(NN, NN0), dtype=self.ftype)
            options['imatrix'] = IM.tocsr()

        self.node = node[:NN]
        cell = cell[:NC]
        self.ds.reinit(NN, cell)


    def coarsen_1(self, isMarkedCell=None, options=None):
        pass



    def grad_lambda(self):
        node = self.node
        cell = self.ds.cell
        NC = self.number_of_cells()
        v0 = node[cell[:, 2], :] - node[cell[:, 1], :]
        v1 = node[cell[:, 0], :] - node[cell[:, 2], :]
        v2 = node[cell[:, 1], :] - node[cell[:, 0], :]
        dim = self.geo_dimension()
        nv = np.cross(v2, -v1)
        Dlambda = np.zeros((NC, 3, dim), dtype=self.ftype)
        if dim == 2:
            length = nv
            W = np.array([[0, 1], [-1, 0]])
            Dlambda[:,0,:] = v0@W/length.reshape((-1, 1))
            Dlambda[:,1,:] = v1@W/length.reshape((-1, 1))
            Dlambda[:,2,:] = v2@W/length.reshape((-1, 1))
        elif dim == 3:
            length = np.sqrt(np.square(nv).sum(axis=1))
            n = nv/length.reshape((-1, 1))
            Dlambda[:,0,:] = np.cross(n, v0)/length.reshape((-1,1))
            Dlambda[:,1,:] = np.cross(n, v1)/length.reshape((-1,1))
            Dlambda[:,2,:] = np.cross(n, v2)/length.reshape((-1,1))
        return Dlambda

    def jacobi_matrix(self, cellidx=None):
        """
        Return
        ------
        J : numpy.array
            `J` is the transpose o  jacobi matrix of each cell.
            The shape of `J` is  `(NC, 2, 2)` or `(NC, 2, 3)`
        """
        node = self.node
        cell = self.ds.cell
        if cellidx is None:
            J = node[cell[:, [1, 2]]] - node[cell[:, [0]]]
        else:
            J = node[cell[cellidx, [1, 2]]] - node[cell[cellidx, [0]]]
        return J

    def rot_lambda(self):
        node = self.node
        cell = self.ds.cell
        NC = self.number_of_cells()
        v0 = node[cell[:, 2], :] - node[cell[:, 1], :]
        v1 = node[cell[:, 0], :] - node[cell[:, 2], :]
        v2 = node[cell[:, 1], :] - node[cell[:, 0], :]
        dim = self.geo_dimension()
        nv = np.cross(v2, -v1)
        Rlambda = np.zeros((NC, 3, dim), dtype=self.ftype)
        if dim == 2:
            length = nv
            Rlambda[:,0,:] = v0/length.reshape((-1, 1))
            Rlambda[:,1,:] = v1/length.reshape((-1, 1))
            Rlambda[:,2,:] = v2/length.reshape((-1, 1))
        elif dim == 3:
            length = np.sqrt(np.square(nv).sum(axis=1))
            Rlambda[:,0,:] = v0/length.reshape((-1, 1))
            Rlambda[:,1,:] = v1/length.reshape((-1, 1))
            Rlambda[:,2,:] = v2/length.reshape((-1, 1))
        return Rlambda

    def area(self, index=None):
        node = self.node
        cell = self.ds.cell
        dim = self.node.shape[1]
        if index is None:
            v1 = node[cell[:, 1], :] - node[cell[:, 0], :]
            v2 = node[cell[:, 2], :] - node[cell[:, 0], :]
        else:
            v1 = node[cell[index, 1], :] - node[cell[index, 0], :]
            v2 = node[cell[index, 2], :] - node[cell[index, 0], :]
        nv = np.cross(v2, -v1)
        if dim == 2:
            a = nv/2.0
        elif dim == 3:
            a = np.sqrt(np.square(nv).sum(axis=1))/2.0
        return a

    def cell_area(self, index=None):
        node = self.node
        cell = self.ds.cell
        dim = self.node.shape[1]
        if index is None:
            v1 = node[cell[:, 1], :] - node[cell[:, 0], :]
            v2 = node[cell[:, 2], :] - node[cell[:, 0], :]
        else:
            v1 = node[cell[index, 1], :] - node[cell[index, 0], :]
            v2 = node[cell[index, 2], :] - node[cell[index, 0], :]
        nv = np.cross(v2, -v1)
        if dim == 2:
            a = nv/2.0
        elif dim == 3:
            a = np.sqrt(np.square(nv).sum(axis=1))/2.0
        return a

    def bc_to_point(self, bc, etype='cell', index=None):
        node = self.node
        entity = self.entity(etype)
        if index is None:
            p = np.einsum('...j, ijk->...ik', bc, node[entity])
        else:
            p = np.einsum('...j, ijk->...ik', bc, node[entity[index]])
        return p
```



##### class LagrangeFiniteElementSpace():

```python
class LagrangeFiniteElementSpace():
    def __init__(self, mesh, p=1, spacetype='C', q=None):
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')
        self.p = p
        if spacetype is 'C':
            if mesh.meshtype is 'interval':
                self.dof = CPLFEMDof1d(mesh, p)
                self.TD = 1
            elif mesh.meshtype is 'tri':
                self.dof = CPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype is 'stri':
                self.dof = CPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype is 'tet':
                self.dof = CPLFEMDof3d(mesh, p)
                self.TD = 3
        elif spacetype is 'D':
            if mesh.meshtype is 'interval':
                self.dof = DPLFEMDof1d(mesh, p)
                self.TD = 1
            elif mesh.meshtype is 'tri':
                self.dof = DPLFEMDof2d(mesh, p)
                self.TD = 2
            elif mesh.meshtype is 'tet':
                self.dof = DPLFEMDof3d(mesh, p)
                self.TD = 3

        if len(mesh.node.shape) == 1:
            self.GD = 1
        else:
            self.GD = mesh.node.shape[1]

        self.spacetype = spacetype
        self.itype = mesh.itype
        self.ftype = mesh.ftype

        q = q if q is not None else p+3
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

        self.mutli_index_matrix = [multi_index_matrix1d, multi_index_matrix2d, multi_index_matrix3d]

    def __str__(self):
        return "Lagrange finite element space!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self):
        return self.dof.cell2dof

    def face_to_dof(self):
        return self.dof.face_to_dof()

    def edge_to_dof(self):
        return self.dof.edge_to_dof()

    def boundary_dof(self, threshold=None):
        if self.spacetype is 'C':
            return self.dof.boundary_dof(threshold=threshold)
        else:
            raise ValueError('This space is a discontinuous space!')

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def grad_recovery(self, uh, method='simple'):
        GD = self.GD
        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()
        ldof = self.number_of_local_dofs()
        p = self.p
        bc = self.dof.multiIndex/p
        guh = uh.grad_value(bc)
        guh = guh.swapaxes(0, 1)
        rguh = self.function(dim=GD)

        if method is 'simple':
            deg = np.bincount(cell2dof.flat, minlength = gdof)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method is 'area':
            measure = self.mesh.entity_measure('cell')
            ws = np.einsum('i, j->ij', measure,np.ones(ldof))
            deg = np.bincount(cell2dof.flat,weights = ws.flat, minlength = gdof)
            guh = np.einsum('ij..., i->ij...',guh,measure)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method is 'distance':
            ipoints = self.interpolation_points()
            bp = self.mesh.entity_barycenter('cell')
            v = bp[:, np.newaxis, :] - ipoints[cell2dof, :]
            d = np.sqrt(np.sum(v**2, axis=-1))
            deg = np.bincount(cell2dof.flat,weights = d.flat, minlength = gdof)
            guh = np.einsum('ij..., ij->ij...',guh,d)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method is 'area_harmonic':
            measure = 1/self.mesh.entity_measure('cell')
            ws = np.einsum('i, j->ij', measure,np.ones(ldof))
            deg = np.bincount(cell2dof.flat,weights = ws.flat, minlength = gdof)
            guh = np.einsum('ij..., i->ij...',guh,measure)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method is 'distance_harmonic':
            ipoints = self.interpolation_points()
            bp = self.mesh.entity_barycenter('cell')
            v = bp[:, np.newaxis, :] - ipoints[cell2dof, :]
            d = 1/np.sqrt(np.sum(v**2, axis=-1))
            deg = np.bincount(cell2dof.flat,weights = d.flat, minlength = gdof)
            guh = np.einsum('ij..., ij->ij...',guh,d)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)
        rguh /= deg.reshape(-1, 1)
        return rguh

    def edge_basis(self, bc, cellidx, lidx, direction=True):
        """
        compute the basis function values at barycentric point bc on edge

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(tdim,)` or `(NQ, tdim)`

        Returns
        -------
        phi : numpy.array
            the shape of 'phi' can be `(NE, ldof)` or `(NE, NQ, ldof)`

        See also
        --------

        Notes
        -----

        """

        mesh = self.mesh

        cell2cell = mesh.ds.cell_to_cell()
        isInEdge = (cell2cell[cellidx, lidx] != cellidx)

        NE = len(cellidx)
        nmap = np.array([1, 2, 0])
        pmap = np.array([2, 0, 1])
        shape = (NE, ) + bc.shape[0:-1] + (3, )
        bcs = np.zeros(shape, dtype=self.mesh.ftype)  # (NE, 3) or (NE, NQ, 3)
        idx = np.arange(NE)

        bcs[idx, ..., nmap[lidx]] = bc[..., 0]
        bcs[idx, ..., pmap[lidx]] = bc[..., 1]

        if direction is False:
            bcs[idx[isInEdge], ..., nmap[lidx[isInEdge]]] = bc[..., 1]
            bcs[idx[isInEdge], ..., pmap[lidx[isInEdge]]] = bc[..., 0]

        return self.basis(bcs)

    def edge_grad_basis(self, bc, cellidx, lidx, direction=True):
        NE = len(cellidx)
        nmap = np.array([1, 2, 0])
        pmap = np.array([2, 0, 1])
        shape = (NE, ) + bc.shape[0:-1] + (3, )
        bcs = np.zeros(shape, dtype=self.mesh.ftype)  # (NE, 3) or (NE, NQ, 3)
        idx = np.arange(NE)
        if direction:
            bcs[idx, ..., nmap[lidx]] = bc[..., 0]
            bcs[idx, ..., pmap[lidx]] = bc[..., 1]
        else:
            bcs[idx, ..., nmap[lidx]] = bc[..., 1]
            bcs[idx, ..., pmap[lidx]] = bc[..., 0]

        p = self.p   # the degree of polynomial basis function
        TD = self.TD

        multiIndex = self.dof.multiIndex

        c = np.arange(1, p+1, dtype=self.itype)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bcs.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bcs[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=self.ftype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]
        ldof = self.number_of_local_dofs()
        shape = bcs.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        Dlambda = self.mesh.grad_lambda()
        gphi = np.einsum('k...ij, kjm->k...im', R, Dlambda[cellidx, :, :])
        return gphi

    def face_basis(self, bc):
        p = self.p   # the degree of polynomial basis function
        TD = self.TD - 1
        multiIndex = self.mutli_index_matrix[TD-1](p)

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi


    def basis(self, bc):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(tdim+1,)` or `(NQ, tdim+1)`
        Returns
        -------
        phi : numpy.array
            the shape of 'phi' can be `(ldof, )` or `(NQ, ldof)`

        See also
        --------

        Notes
        -----

        """
        p = self.p   # the degree of polynomial basis function

        if p == 0 and self.spacetype == 'D':
            if len(bc.shape) == 1:
                return np.ones(1, dtype=self.ftype)
            else:
                return np.ones((bc.shape[0], 1), dtype=self.ftype)

        TD = self.TD
        multiIndex = self.dof.multiIndex

        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi

    def grad_basis(self, bc, cellidx=None):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.array
            the shape of `bc` can be `(tdim+1,)` or `(NQ, tdim+1)`

        Returns
        -------
        gphi : numpy.array
            the shape of `gphi` can b `(NC, ldof, gdim)' or
            `(NQ, NC, ldof, gdim)'

        See also
        --------

        Notes
        -----

        """
        p = self.p   # the degree of polynomial basis function
        TD = self.TD

        multiIndex = self.dof.multiIndex

        c = np.arange(1, p+1, dtype=self.itype)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=self.ftype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]
        ldof = self.number_of_local_dofs()
        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        Dlambda = self.mesh.grad_lambda()
        if cellidx is None:
            gphi = np.einsum('...ij, kjm->...kim', R, Dlambda)
        else:
            gphi = np.einsum('...ij, kjm->...kim', R, Dlambda[cellidx, :, :])
        return gphi

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...j, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, phi, uh[cell2dof])
        else:
            val = np.einsum(s1, phi, uh[cell2dof[cellidx]])
        return val

    def grad_value(self, uh, bc, cellidx=None):
        gphi = self.grad_basis(bc, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        if self.geo_dimension() == 1:
            s1 = '...ijm, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        else:
            s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])

        if cellidx is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[cellidx]])
        return val

    def div_value(self, uh, bc, cellidx=None):
        dim = len(uh.shape)
        gdim = self.geo_dimension()
        if (dim == 2) & (uh.shape[1] == gdim):
            val = self.grad_value(uh, bc, cellidx=cellidx)
            return val.trace(axis1=-2, axis2=-1)
        else:
            raise ValueError("The shape of uh should be (gdof, gdim)!")

    def interpolation(self, u, dim=None):
        ipoint = self.dof.interpolation_points()
        uI = u(ipoint)
        return self.function(dim=dim, array=uI)

    def projection(self, u):
        """
        """
        M= self.mass_matrix()
        F = self.source_vector(u)
        uh = self.function()
        uh[:] = spsolve(M, F).reshape(-1)
        return uh

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in [None, 1]:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)

    def linear_elasticity_matrix(self, mu, lam):
        """
        construct the linear elasticity fem matrix
        """
        cellmeasure = self.cellmeasure
        cell2dof = self.cell_to_dof()
        GD = self.GD

        qf = self.integrator
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad = self.grad_basis(bcs)

        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()

        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)

        if GD == 2:
            idx = [(0, 0), (0, 1),  (1, 1)]
            imap = {(0, 0):0, (0, 1):1, (1, 1):2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0):0, (0, 1):1, (0, 2):2, (1, 1):3, (1, 2):4, (2, 2):5}
        A = []
        for k, (i, j) in enumerate(idx):
            Aij = np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., i], grad[..., j], cellmeasure)
            A.append(csr_matrix((Aij.flat, (I.flat, J.flat)), shape=(gdof, gdof)))

        T = csr_matrix((gdof, gdof), dtype=self.ftype)
        D = csr_matrix((gdof, gdof), dtype=self.ftype)
        C = []
        for i in range(GD):
            D += A[imap[(i, i)]]
            C.append([T]*GD)
        D *= mu

        for i in range(GD):
            for j in range(i, GD):
                if i == j:
                    C[i][j] = D + (mu+lam)*A[imap[(i, i)]]
                else:
                    C[i][j] = lam*A[imap[(i, j)]] + mu*A[imap[(i, j)]].T
                    C[j][i] = C[i][j].T
        A = bmat(C, format='csr') # format = bsr ??
        return A

    def stiff_matrix(self, cfun=None):
        p = self.p
        GD = self.geo_dimension()

        if p == 0:
            raise ValueError('The space order is 0!')

        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        gphi = self.grad_basis(bcs)

        if cfun is not None:
            ps = self.mesh.bc_to_point(bcs)
            d = cfun(ps)

            if isinstance(d, (int, float)):
                dgphi = d*gphi
            elif len(d) == GD:
                dgphi = np.einsum('m, ...im->...im', d, gphi)
            elif isinstance(d, np.ndarray):
                if len(d.shape) == 1:
                    dgphi = np.einsum('i, ...imn->...imn', d, gphi)
                elif len(d.shape) == 2:
                    dgphi = np.einsum('...i, ...imn->...imn', d, gphi)
                elif len(d.shape) == 3: #TODO:
                    dgphi = np.einsum('...imn, ...in->...im', d, gphi)
                elif len(d.shape) == 4: #TODO:
                    dgphi = np.einsum('...imn, ...in->...im', d, gphi)
                else:
                    raise ValueError("The ndarray shape length should < 5!")
            else:
                raise ValueError(
                        "The return of cfun is not a number or ndarray!"
                        )
        else:
            dgphi = gphi

        # Compute the element sitffness matrix
        # ws:(NQ,)
        # dgphi: (NQ, NC, ldof, GD)
        A = np.einsum('i, ijkm, ijpm, j->jkp',
                ws, dgphi, gphi, self.cellmeasure,
                optimize=True)
        cell2dof = self.cell_to_dof()
        ldof = self.number_of_local_dofs()
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs()

        # Construct the stiffness matrix
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def mass_matrix(self, cfun=None, barycenter=False):
        p = self.p
        mesh = self.mesh
        cellmeasure = self.cellmeasure

        if p == 0:
            NC = mesh.number_of_cells()
            M = spdiags(cellmeasure, 0, NC, NC)
            return M

        # bcs: (NQ, TD+1)
        # ws: (NQ, )
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        # phi: (NQ, ldof)
        phi = self.basis(bcs)

        if cfun is not None:
            if barycenter is True:
                d = cfun(bcs) # (NQ, NC)
            else:
                ps = self.mesh.bc_to_point(bcs) # (NQ, NC, GD)
                d = cfun(ps) # (NQ, NC)

            if isinstance(d, (int, float)):
                dphi = d*phi
            elif isinstance(d, np.ndarray):
                if (len(d.shape) == 1):
                    dphi = np.einsum('i, mj->mij', d, phi)
                elif len(d.shape) == 2:
                    dphi = np.einsum('mi, mj->mij', d, phi)
                else:
                    raise ValueError("The ndarray shape length should < 3!")
            else:
                raise ValueError(
                        "The return of cfun is not a number or ndarray!"
                        )
        else:
            dphi = phi

        if len(dphi.shape) == 2:
            M = np.einsum(
                    'm, mj, mk, i->ijk',
                    ws, dphi, phi, cellmeasure,
                    optimize=True)
        elif len(dphi.shape) == 3:
            M = np.einsum(
                    'm, mij, mk, i->ijk',
                    ws, dphi, phi, self.cellmeasure,
                    optimize=True)

        cell2dof = self.cell_to_dof()
        ldof = self.number_of_local_dofs()
        I = np.einsum('ij, k->ijk',  cell2dof, np.ones(ldof))
        J = I.swapaxes(-1, -2)

        gdof = self.number_of_global_dofs()
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M

    def source_vector(self, f, dim=None):
        p = self.p
        cellmeasure = self.cellmeasure
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        pp = self.mesh.bc_to_point(bcs)
        fval = f(pp)

        if p > 0:
            phi = self.basis(bcs)
            # bb: (NC, ldof)
            bb = np.einsum('m, mi..., mk, i->ik...',
                    ws, fval, phi, self.cellmeasure)
            cell2dof = self.cell_to_dof() #(NC, ldof)
            gdof = self.number_of_global_dofs()
            shape = gdof if dim is None else (gdof, dim)
            b = np.zeros(shape, dtype=self.ftype)
            if dim is None:
                np.add.at(b, cell2dof, bb)
            else:
                np.add.at(b, (cell2dof, np.s_[:]), bb)
        else:
            b = np.einsum('i, ik..., k->k...', ws, fval, cellmeasure)
        return b.T.flat

    def set_dirichlet_bc(self, uh, g, is_dirichlet_boundary=None):
        """
        初始化解 uh  的第一类边界条件。
        """
        ipoints = self.interpolation_points()
        isDDof = self.boundary_dof(threshold=is_dirichlet_boundary)
        uh[isDDof] = g(ipoints[isDDof])
        return isDDof

    def to_function(self, data):
        p = self.p
        if p == 1:
            uh = self.function(array=data)
            return uh
        elif p == 2:
            cell2dof = self.cell_to_dof()
            uh = self.function()
            uh[cell2dof] = data[:, [0, 5, 4, 1, 3, 2]]
            return uh
```



##### class PoissonFEMModel(object):

```python
class PoissonFEMModel(object):
    def __init__(self, pde, mesh, p, q=3):
        self.space = LagrangeFiniteElementSpace(mesh, p, q=q)
        self.mesh = self.space.mesh
        self.pde = pde
        self.uh = self.space.function()
        self.uI = self.space.interpolation(pde.solution)
        self.integrator = self.mesh.integrator(p+2)
    def recover_estimate(self, rguh):
        qf = self.integrator
        bcs, ws = qf.quadpts, qf.weights

        val0 = rguh.value(bcs)
        val1 = self.uh.grad_value(bcs)
        l = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, l)
        e *= self.space.cellmeasure
        return np.sqrt(e)

    def residual_estimate(self, uh=None):
        if uh is None:
            uh = self.uh
        mesh = self.mesh
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()

        n = mesh.face_normal()
        bc = np.array([1/(GD+1)]*(GD+1), dtype=mesh.ftype)
        grad = uh.grad_value(bc)

        ps = mesh.bc_to_point(bc)
        try:
            d = self.pde.diffusion_coefficient(ps)
        except AttributeError:
            d = np.ones(NC, dtype=mesh.ftype)

        if isinstance(d, float):
            grad *= d
        elif len(d) == GD:
            grad = np.einsum('m, im->im', d, grad)
        elif isinstance(d, np.ndarray):
            if len(d.shape) == 1:
                grad = np.einsum('i, im->im', d, grad)
            elif len(d.shape) == 2:
                grad = np.einsum('im, im->im', d, grad)
            elif len(d.shape) == 3:
                grad = np.einsum('imn, in->in', d, grad)

        if GD == 2:
            face2cell = mesh.ds.edge_to_cell()
            h = np.sqrt(np.sum(n**2, axis=-1))
        elif GD == 3:
            face2cell = mesh.ds.face_to_cell()
            h = np.sum(n**2, axis=-1)**(1/4)

        J = h*np.sum((grad[face2cell[:, 0]] - grad[face2cell[:, 1]])*n, axis=-1)**2

        NC = mesh.number_of_cells()
        eta = np.zeros(NC, dtype=mesh.ftype)
        np.add.at(eta, face2cell[:, 0], J)
        np.add.at(eta, face2cell[:, 1], J)
        return np.sqrt(eta)

    def get_left_matrix(self):
        return self.space.stiff_matrix()

    def get_right_vector(self):
        return self.space.source_vector(self.pde.source)

    def solve(self):
        bc = DirichletBC(self.space, self.pde.dirichlet)

        start = timer()
        A = self.get_left_matrix()
        b = self.get_right_vector()
        end = timer()
        self.A = A

        print("Construct linear system time:", end - start)

        AD, b = bc.apply(A, b)

        start = timer()
        self.uh[:] = spsolve(AD, b)
        end = timer()
        print("Solve time:", end-start)

        ls = {'A': AD, 'b': b, 'solution': self.uh.copy()}

        return ls  # return the linear system

    def l2_error(self):
        e = self.uh - self.uI
        return np.sqrt(np.mean(e**2))

    def uIuh_error(self):
        e = self.uh - self.uI
        return np.sqrt(e@self.A@e)

    def L2_error(self, uh=None):
        u = self.pde.solution
        if uh is None:
            uh = self.uh.value
        else:
            uh = uh.value
        L2 = self.space.integralalg.L2_error(u, uh)
        return L2

    def H1_semi_error(self, uh=None):
        gu = self.pde.gradient
        if uh is None:
            guh = self.uh.grad_value
        else:
            guh = uh.grad_value
        H1 = self.space.integralalg.L2_error(gu, guh)
        return H1

    def recover_error(self, rguh):
        gu = self.pde.gradient
        guh = rguh.value
        mesh = self.mesh
        re = self.space.integralalg.L2_error(gu, guh, mesh)
        return re
```



##### end