# -*- coding: utf-8 -*-
r"""
EkEkStar

AUTHORS:

 - Milton Minervino, 2017, initial version
 - Sébastien Labbé, July 6th 2017: added doctests, package, improve object
   oriented structure of the classes

.. TODO::

    - Clean method kFace._plot(geosub) so that it takes projection
      information (like vectors or matrix or ...) instead of the geosub

    - Patch should contain the multiplicity, not the faces. 
    
    - Fix the addition of patchs (linear time instead of quadratic time).

    - Use rainbow() or something else to automatically choose colors

    - Allow the user to choose the colors of faces

    - input dual should be a bool True or False not 0 or 1
 
EXAMPLES::

    sage: from EkEkstar import GeoSub, kPatch, kFace
    sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
    sage: geosub = GeoSub(sub,2,1,1)
    sage: P = kPatch([kFace((0,0,0),(1,2),d=1),
    ....:             kFace((0,0,1),(1,3),d=1),
    ....:             kFace((0,1,0),(2,1),d=1),
    ....:             kFace((0,0,0),(3,1),d=1)])
    sage: Q = geosub(P, 6)
    sage: Q
    Patch of 32 faces
    sage: _ = Q.plot(geosub)

REMAINDER:

This is a remainder for how to get the import statements::

    sage: import_statements('Graphics')
    from sage.plot.graphics import Graphics
"""
from itertools import product, combinations
from sage.misc.cachefunc import cached_method
from sage.structure.sage_object import SageObject
from sage.rings.integer_ring import ZZ
from sage.modules.free_module_element import vector, zero_vector
from sage.groups.perm_gps.permgroup_named import SymmetricGroup
from sage.rings.all import CC
from sage.combinat.words.morphism import WordMorphism
from sage.rings.number_field.number_field import NumberField
from sage.plot.colors import Color
from sage.plot.graphics import Graphics
from sage.plot.polygon import polygon2d
from sage.plot.line import line

##########
#  Classes
##########
class kFace(SageObject):
    r"""
    EXAMPLES:

    Face based at (0,0,0) of type (1,2)::

        sage: from EkEkstar import kFace
        sage: F = kFace((0,0,0),(1,2))
        sage: F
        +[(0, 0, 0), (1, 2)]
        
    Face based at (0,0,0) of type (3,1): the type is changed to (1,3) 
    and the multiplicity of the face turns to -1::

        sage: kFace((0,0,0),(3,1))
        -[(0, 0, 0), (1, 3)]
        
    Dual face based at (0,0,0,0) of type (1) with multiplicity -3::
        
        sage: kFace((0,0,0,0),(1),m=-3,d=1)
        -3[(0, 0, 0, 0), 1]*
    """
    def __init__(self, v, t, m=1, d=0, color=None):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace
            sage: F = kFace((0,0,0),(1,2))
            sage: F
            +[(0, 0, 0), (1, 2)]
        """
        from sage.groups.perm_gps.permgroup_named import SymmetricGroup
        from sage.functions.generalized import sign

        self._vector = (ZZ**len(v))(v)
        self._vector.set_immutable()
        #if not((t in ZZ) and 1 <= t <= len(v)):
        #    raise ValueError('The type must be an integer between 1 and len(v)')
        if (t in ZZ):
            self._type = t
            self._mult = m
        else:
            t1 = list(t)
            t1.sort()
            self._type = tuple(t1)
            if all(t1[i] < t1[i+1] for i in range(len(t1)-1)):
                L = [sign(p) for p in SymmetricGroup(len(self._type)) if p(list(t)) == t1]
                self._mult = L[0]*m
            else:
                self._mult = 0
        
        self._dual = d
        
        if color is None:
            if self._type == (1,):
                self._color = Color((1,0,0))
            elif self._type == (2,):
                self._color = Color((0,0,1))
            elif self._type == (3,):
                self._color = Color((0,1,0))
            elif self._type == (4,):
                self._color = Color('yellow')
            elif self._type == (5,):
                self._color = Color('purple')    
            elif self._type == (1,2):
                self._color = Color('red')  
            elif self._type == (1,3):
                self._color = Color('blue')   
            elif self._type == (1,4):
                self._color = Color('green')
            elif self._type == (1,5):
                self._color = Color('purple')     
            elif self._type == (2,3):
                self._color = Color('orange') 
            elif self._type == (2,4):
                self._color = Color('pink')
            elif self._type == (2,5):
                self._color = Color('brown')     
            elif self._type == (3,4):
                self._color = Color('yellow')  
            elif self._type == (3,5):
                self._color = Color('darkgreen')
            elif self._type == (4,5):
                self._color = Color('lightblue') 
            elif self._type == (1,2,3):
                self._color = Color('red')  
            elif self._type == (1,2,4):
                self._color = Color('blue')   
            elif self._type == (1,2,5):
                self._color = Color('green')
            elif self._type == (1,3,4):
                self._color = Color('purple')     
            elif self._type == (1,3,5):
                self._color = Color('orange') 
            elif self._type == (1,4,5):
                self._color = Color('pink')
            elif self._type == (2,3,4):
                self._color = Color('brown')     
            elif self._type == (2,3,5):
                self._color = Color('yellow')  
            elif self._type == (2,4,5):
                self._color = Color('darkgreen')
            elif self._type == (3,4,5):
                self._color = Color('lightblue')  
            #else:
            #    print(self._type)
            #    lol = RR(len(self._type)+1)
            #    self._color = Color(self._type[0]/lol,self._type[1]/lol,self._type[2]/lol)                                            
        else:
            self._color = color

    def __repr__(self):
        r"""
        String representation.
        """
        if self.dual() == 1:
            if self.mult() == -1:
                return "%s[%s, %s]*"%('-',self.vector(), self.type())
            elif self.mult() == 1:
                return "%s[%s, %s]*"%('+',self.vector(), self.type())
            elif self.mult() == 0:
                return "[]*"
            else:
                return "%s[%s, %s]*"%(self.mult(),self.vector(), self.type())
        else:
            if self.mult() == -1:
                return "%s[%s, %s]"%('-',self.vector(), self.type())
            elif self.mult() == 1:
                return "%s[%s, %s]"%('+',self.vector(), self.type())
            elif self.mult() == 0:
                return "[]"
            else:
                return "%s[%s, %s]"%(self.mult(),self.vector(), self.type())

    def __eq__(self, other):
        return (isinstance(other, kFace) and
                self.vector() == other.vector() and
                self.type() == other.type() and 
                self.mult() == other.mult() and 
                self.dual() == other.dual())

    @cached_method
    def __hash__(self):
        return hash((self.vector(), self.type(), self.mult(), self.dual()))

    def __add__(self, other):
        r"""
        EXAMPLES::
        
            sage: from EkEkstar import kFace
            sage: kFace((0,0,0),(1,3)) + kFace((0,0,0),(3,1))
            Patch: []
            sage: kFace((0,0,0),(1,3)) + kFace((0,0,0),(1,3))
            Patch: [2[(0, 0, 0), (1, 3)]]

        """
        if isinstance(other, kFace):
            return kPatch([self, other])
        else:
            return kPatch(other).union(self)

    def vector(self):
        return self._vector

    def type(self):
        return self._type
        
    def mult(self):
        return self._mult

    def dual(self):
        return self._dual

    def color(self, color=None):
        if color is None:
            return self._color
        else:
            self._color = Color(color)
            
    def _plot(self, geosub):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace, GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: geosub = GeoSub(sub,2,1,1)
            sage: f = kFace((0,0,0),(1,2),d=1)
            sage: _ = f._plot(geosub)

        ::

            sage: sub = {1:[1,2,3,3,3,3], 2:[1,3], 3:[1]}
            sage: geosub = GeoSub(sub,2,1,1)
            sage: f = kFace((0,0,0),(1,2),d=1)
            sage: _ = f._plot(geosub)
        """
        v = self.vector()
        t = self.type()
        col = self._color
        G = Graphics()
        
        K = geosub.field()
        b = K.gen()
                
        num = geosub._sigma_dict.keys()
        
        if self._dual == 1:
            h = list(set(num)-set(t))
            B = b
            vec = geosub.dominant_left_eigenvector()
            emb = geosub.contracting_eigenvalues_indices()
        else:
            h = list(t)
            B = b**(-1) 
            vec = -geosub.dominant_left_eigenvector() 
            emb = geosub.dilating_eigenvalues_indices() 
        
        el = v*vec
        iter = 0

        conjugates = geosub.complex_embeddings()

        if len(h) == 1:
            if conjugates[emb[0]].is_real() == True:
                bp = zero_vector(CC, len(emb))
                for i in range(len(emb)):
                    bp[i] = K(el).complex_embeddings()[emb[i]]
                bp1 = zero_vector(CC, len(emb))
                for i in range(len(emb)):
                    bp1[i] = K((el+vec[h[0]-1])).complex_embeddings()[emb[i]] 
                if len(emb) == 1:
                    return line([bp[0],bp1[0]],color = col,thickness = 3) 
                else: 
                    return line([bp,bp1],color = col,thickness = 3)      
            else: 
                bp = K(el).complex_embeddings()[emb[0]]
                bp1 = K((el+vec[h[0]-1])).complex_embeddings()[emb[0]]
                return line([bp,bp1],color = col,thickness = 3)
        elif len(h) == 2: 
            if conjugates[emb[0]].is_real() == True:
                bp = (  K(el).complex_embeddings()[emb[0]], 
                        K(el).complex_embeddings()[emb[1]])
                bp1 = ( K(el+vec[h[0]-1]).complex_embeddings()[emb[0]],
                        K(el+vec[h[0]-1]).complex_embeddings()[emb[1]] )
                bp2 = ( K(el+vec[h[0]-1]+vec[h[1]-1]).complex_embeddings()[emb[0]],
                        K(el+vec[h[0]-1]+vec[h[1]-1]).complex_embeddings()[emb[1]])
                bp3 = ( K(el+vec[h[1]-1]).complex_embeddings()[emb[0]],
                        K(el+vec[h[1]-1]).complex_embeddings()[emb[1]])
                return polygon2d([bp,bp1,bp2,bp3],color=col,thickness=.1,alpha = 0.8)
            else:   
                bp =  K(el).complex_embeddings()[emb[0]]
                bp1 = K(el+vec[h[0]-1]).complex_embeddings()[emb[0]]
                bp2 = K(el+vec[h[0]-1]+vec[h[1]-1]).complex_embeddings()[emb[0]]
                bp3 = K(el+vec[h[1]-1]).complex_embeddings()[emb[0]]
                return polygon2d([[bp[0],bp[1]],[bp1[0],bp1[1]],[bp2[0],bp2[1]],[bp3[0],bp3[1]]],color=col,thickness=.1,alpha = 0.8)     
            
        else:
            raise NotImplementedError("Plotting is implemented only for patches in two or three dimensions.")
        return G


class kPatch(SageObject):
    def __init__(self, faces, face_contour=None):
        L = [kFace(f.vector(), f.type(), f.mult(), f.dual(), f.color()) for f in faces if (isinstance(f,kFace) and f.mult() != 0)]
        #self._faces = frozenset(kFace(x.vector(),x.type(),L.count(x), x.dual(), x.color()) for x in set(L)) 
        #[(x,L.count(x)) for x in set(L)]
        
        L2 = []
        for x in L:
            L1 = [y for y in L if (y.vector(),y.type()) == (x.vector(),x.type())]
            s = sum([L.count(x)*x.mult() for x in set(L1)])
            if s != 0:
                L2 += [kFace(x.vector(),x.type(),s,x.dual(),x.color())] 
        self._faces = set(L2)
        
        try:
            f0 = next(iter(self._faces))
        except StopIteration:
            self._dimension = None
        else:
            self._dimension = len(f0.vector())
        if not face_contour is None:
            self._face_contour = face_contour
        else:
            self._face_contour = {
                    1: [vector(_) for _ in [(0,0,0),(0,1,0),(0,1,1),(0,0,1)]],
                    2: [vector(_) for _ in [(0,0,0),(0,0,1),(1,0,1),(1,0,0)]],
                    3: [vector(_) for _ in [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]]
            }
            
    def __len__(self):
        return len(self._faces)

    def __iter__(self):
        return iter(self._faces)
       
    def __add__(self, other):
        return self.union(other)
        
    def dimension(self):
        return self._dimension

    def __repr__(self):
        if len(self) <= 30:
            L = list(self)
            #L.sort(key=lambda x : (x.vector(),x.type(),x.mult(),x.dual()))
            return "Patch: %s"%L
        else:
            return "Patch of %s faces"%len(self)
        
    def union(self, other):
        if isinstance(other, kFace):
            return kPatch(list(self._faces) + [other])
        elif isinstance(other, kPatch):
            return kPatch(list(self._faces) + list(other._faces))
        else:
            return kPatch(list(self._faces) + other)
        
    def plot(self, geosub):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace, kPatch, GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: geosub = GeoSub(sub,2,1,1)
            sage: P = kPatch([kFace((0,0,0),(1,2),d=1),
            ....:             kFace((0,0,1),(1,3),d=1),
            ....:             kFace((0,1,0),(2,1),d=1),
            ....:             kFace((0,0,0),(3,1),d=1)])
            sage: _ = P.plot(geosub)
        """
        G = Graphics()
        for face in self:
            G += face._plot(geosub)
        G.set_aspect_ratio(1)
        return G
              
        #if self.dimension() == 2:
        #    G = Graphics()
        #    for face in self:
        #        G += face._plot(None, None, 1)
        #    G.set_aspect_ratio(1)
        #    return G
        #if self.dimension() == 3:
        #    if projmat is None:
        #        projmat = matrix(2, [-1.7320508075688772*0.5, 1.7320508075688772*0.5, 0, -0.5, -0.5, 1])
            
     
             

def psauto(sub, ps):
    d = {}
    v = sub.values()
    for i in range(len(v)):
        L = []
        for j in range(len(v[i])):
            if ps == 1:
                L.append((v[i][j],sub[i+1][0:j]))
            elif ps == 0:
                L.append((v[i][j],sub[i+1][j+1:len(v[i])])) 
        d[i+1] = L  
    return d             
                  

def invauto(sub,ps):
    d = {}
    k = sub.keys()
    Gr = psauto(sub,ps)
    for a in k:
        L = []
        for i in k:
            L += [(i,Gr[i][j][1]) for j in range(len(sub[i])) if sub[i][j] == a]
        d[a] = L    
    return d         
      

def abelian(L, alphabet):
    r"""
    EXAMPLES::

        sage: from EkEkstar.EkEkstar import abelian
        sage: abelian([1,0,1,2,3,1,1,2,2], [0,1,2,3])
        (1, 4, 3, 1)
    """
    return vector([L.count(i) for i in alphabet])

class GeoSub(SageObject):
    r"""
    EXAMPLES::

        sage: from EkEkstar import GeoSub
        sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
        sage: E = GeoSub(sub,2,1,0)
        sage: E
        E_2(1->12, 2->13, 3->1)
    """
    def __init__(self, sigma, k, presuf, dual):
        self._sigma_dict = sigma
        self._sigma = WordMorphism(sigma)
        self._k = k
        self._presuf = presuf
        self._dual = dual

    @cached_method
    def field(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2,1,0)
            sage: E.field()
            Number Field in b with defining polynomial x^3 - x^2 - x - 1
        """
        M = self._sigma.incidence_matrix()
        b1 = max(M.eigenvalues())
        f = b1.minpoly()
        K = NumberField(f, 'b')
        return K
        
    def gen(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2,1,0)
            sage: E.gen()
            b^2 - b - 1
        """
        b = self.field().gen()
        if self._dual == 1:
            return b
        else:
            return b**-1
        
    def dominant_left_eigenvector(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2,1,0)
            sage: E.dominant_left_eigenvector()
            (1, b - 1, b^2 - b - 1)
        """
        M = self._sigma.incidence_matrix()-self.field().gen()
        return M.left_kernel().basis()[0]
        
    def dominant_right_eigenvector(self):
        r"""
        EXAMPLES::
            
            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2,1,0)
            sage: E.dominant_right_eigenvector()
            (1, b^2 - b - 1, -b^2 + 2*b)
        """
        M = self._sigma.incidence_matrix()-self.field().gen()
        return M.right_kernel().basis()[0]

    def complex_embeddings(self):
        r"""
        EXAMPLES::
            
            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2,1,0)
            sage: E.complex_embeddings()
            [-0.419643377607081 - 0.606290729207199*I,
             -0.419643377607081 + 0.606290729207199*I,
             1.83928675521416]
        """
        return self.field().gen().complex_embeddings()

    def contracting_eigenvalues_indices(self):
        r"""
        EXAMPLES::
            
            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2,1,0)
            sage: E.contracting_eigenvalues_indices()
            [0, 1]
        """
        L = self.complex_embeddings()
        return [L.index(x) for x in L if abs(x)<1]

    def dilating_eigenvalues_indices(self):
        r"""
        EXAMPLES::
            
            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2,1,0)
            sage: E.dilating_eigenvalues_indices()
            [2]
        """
        L = self.complex_embeddings()
        return [L.index(x) for x in L if abs(x)>1]

    @cached_method
    def base_iter(self):
        r"""
        EXAMPLES::
            
            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2,1,0)
            sage: E.base_iter()
            {(1, 2): [[(0, 0, 0), (1,)],
            [(1, 0, 0), (2,)],
            [(0, 0, 0), (1, 1)],
            [(1, 0, 0), (1, 3)],
            [(1, 0, 0), (2, 1)],
            [(2, 0, 0), (2, 3)]],
            (1, 3): [[(0, 0, 0), (1,)],
            [(1, 0, 0), (2,)],
            [(0, 0, 0), (1, 1)],
            [(1, 0, 0), (2, 1)]],
            (2, 3): [[(0, 0, 0), (1,)],
            [(1, 0, 0), (3,)],
            [(0, 0, 0), (1, 1)],
            [(1, 0, 0), (3, 1)]]}
        """
        X = {}
        S = self._sigma_dict.keys()
        for x in combinations(S,self._k):
            X[x] = []
            bigL = []
            for y in x:
                if self._dual == 0:
                    bigL.append(psauto(self._sigma_dict,self._presuf)[y])
                else:
                    bigL.append(invauto(self._sigma_dict,self._presuf)[y])
                Lpro = list(product(*bigL))
                for el in Lpro:
                    z = []
                    w = []
                    for i in range(len(el)):
                        z += el[i][1]
                        w.append(el[i][0])
                    if self._dual == 0:
                        X[x].append([abelian(z, S),tuple(w)])
                    else:
                        M = self._sigma.incidence_matrix()
                        X[x].append([-M.inverse()*abelian(z, S),tuple(w)])
        return X
           
    def __call__(self, patch, iterations=1):
        r"""
        EXAMPLES::

            sage: from EkEkstar import GeoSub, kPatch, kFace
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: geosub = GeoSub(sub,2,1,1)
            sage: P = kPatch([kFace((0,0,0),(1,2),d=1),
            ....:             kFace((0,0,1),(1,3),d=1),
            ....:             kFace((0,1,0),(2,1),d=1),
            ....:             kFace((0,0,0),(3,1),d=1)])
            sage: Q = geosub(P, 6)
            sage: Q
            Patch of 32 faces
        """
        if iterations == 0:
            return kPatch(patch)
        elif iterations < 0:
            raise ValueError("iterations (=%s) must be >= 0." % iterations)
        else:
            old_faces = patch
            for i in range(iterations):
                new_faces = []
                for f in old_faces:
                    if f.mult() != 0:
                        new_faces.extend(self._call_on_face(f, color=f.color()))
                old_faces = new_faces
            return kPatch(new_faces) 
    def matrix(self):
        if self._dual == 0:
            return self._sigma.incidence_matrix()
        else:
            return self._sigma.incidence_matrix().inverse()
        
    def _call_on_face(self, face, color=None):
        x_new = self.matrix() * face.vector()
        t = face.type()
        D = self._dual
        if D == 1:
            return (kFace(x_new + (-1)**(self._dual)*y1, y2, (-1)**(sum(t)+sum(y2))*face.mult(), D)                     for (y1, y2) in self.base_iter()[t] if len(y2) == self._k)
        else:
            return (kFace(x_new + (-1)**(self._dual)*y1, y2, face.mult(), D) 
                    for (y1, y2) in self._base_iter()[t] if len(y2) == self._k)
                
    def __repr__(self):
        r"""
        EXAMPLES::
            
            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: GeoSub(sub, 2, 1, 0)
            E_2(1->12, 2->13, 3->1)
            sage: GeoSub(sub, 2, 1, 1)
            E*_2(1->12, 2->13, 3->1)
        """
        if self._dual == 0: 
            return "E_%s(%s)" % (self._k,str(self._sigma))
        else: 
            return "E*_%s(%s)" % (self._k,str(self._sigma))
        


