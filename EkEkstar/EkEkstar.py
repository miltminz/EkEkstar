# -*- coding: utf-8 -*-
r"""
EkEkStar

AUTHORS:

 - Milton Minervino, 2017, initial version
 - Sébastien Labbé, July 6th 2017: added doctests, package, improve object
   oriented structure of the classes, multiplicity stored in the patch not
   in the faces. Fixed the creation of patchs (linear time instead of
   quadratic time). Added a dozen of doctests.

.. TODO::

    - Clean method kFace._plot(geosub) so that it takes projection
      information (like vectors or matrix or ...) instead of the geosub

    - Fix some proper ordering for the faces (problems with doctests).

    - Use rainbow() or something else to automatically choose colors

    - Allow the user to choose the colors of faces

    - input dual should be a bool True or False not 0 or 1

EXAMPLES::

    sage: from EkEkstar import GeoSub, kPatch, kFace
    sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
    sage: geosub = GeoSub(sub,2, 'prefix',1)
    sage: P = kPatch([kFace((0,0,0),(1,2),d=1),
    ....:             kFace((0,0,1),(1,3),d=1),
    ....:             kFace((0,1,0),(2,1),d=1),
    ....:             kFace((0,0,0),(3,1),d=1)])
    sage: Q = geosub(P, 6)
    sage: Q
    Patch of 47 faces
    sage: _ = Q.plot(geosub)

REMAINDER:

This is a remainder for how to get the import statements::

    sage: import_statements('Graphics')
    from sage.plot.graphics import Graphics
"""
from itertools import product, combinations
from collections import Counter
from numpy import argsort
from sage.misc.cachefunc import cached_method
from sage.structure.sage_object import SageObject
from sage.rings.integer_ring import ZZ
from sage.modules.free_module_element import vector, zero_vector
from sage.rings.all import CC
from sage.combinat.words.morphism import WordMorphism
from sage.combinat.permutation import Permutation
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
        [(0, 0, 0), (1, 2)]
        
    Face based at (0,0,0) of type (3,1): the type is changed to (1,3) 
    and the multiplicity of the face turns to -1::

        sage: kFace((0,0,0),(3,1))
        [(0, 0, 0), (3, 1)]
        
    Dual face based at (0,0,0,0) of type (1) with multiplicity -3::
        
        sage: kFace((0,0,0,0),(1),d=1)
        [(0, 0, 0, 0), 1]*
    """
    def __init__(self, v, t, d=0, color=None):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace
            sage: F = kFace((0,0,0),(1,2))
            sage: F
            [(0, 0, 0), (1, 2)]
        """
        self._vector = (ZZ**len(v))(v)
        self._vector.set_immutable()
        #if not((t in ZZ) and 1 <= t <= len(v)):
        #    raise ValueError('The type must be an integer between 1 and len(v)')

        if t in ZZ:
            self._type = t
        else:
            self._type = t
        
        self._dual = d

        self._color = color
        


    def vector(self):
        return self._vector

    def type(self):
        return self._type

    def dual(self):
        return self._dual

    def color(self, color=None):
        if color is not None:
            self._color = Color(color)
            return
        if self._color is None:
            sorted_type = self.sorted_type()
            if sorted_type == (1,):
                self._color = Color((1,0,0))
            elif sorted_type == (2,):
                self._color = Color((0,0,1))
            elif sorted_type == (3,):
                self._color = Color((0,1,0))
            elif sorted_type == (4,):
                self._color = Color('yellow')
            elif sorted_type == (5,):
                self._color = Color('purple')    
            elif sorted_type == (1,2):
                self._color = Color('red')  
            elif sorted_type == (1,3):
                self._color = Color('blue')   
            elif sorted_type == (1,4):
                self._color = Color('green')
            elif sorted_type == (1,5):
                self._color = Color('purple')     
            elif sorted_type == (2,3):
                self._color = Color('orange') 
            elif sorted_type == (2,4):
                self._color = Color('pink')
            elif sorted_type == (2,5):
                self._color = Color('brown')     
            elif sorted_type == (3,4):
                self._color = Color('yellow')  
            elif sorted_type == (3,5):
                self._color = Color('darkgreen')
            elif sorted_type == (4,5):
                self._color = Color('lightblue') 
            elif sorted_type == (1,2,3):
                self._color = Color('red')  
            elif sorted_type == (1,2,4):
                self._color = Color('blue')   
            elif sorted_type == (1,2,5):
                self._color = Color('green')
            elif sorted_type == (1,3,4):
                self._color = Color('purple')     
            elif sorted_type == (1,3,5):
                self._color = Color('orange') 
            elif sorted_type == (1,4,5):
                self._color = Color('pink')
            elif sorted_type == (2,3,4):
                self._color = Color('brown')     
            elif sorted_type == (2,3,5):
                self._color = Color('yellow')  
            elif sorted_type == (2,4,5):
                self._color = Color('darkgreen')
            elif sorted_type == (3,4,5):
                self._color = Color('lightblue')  
            #else:
            #    print(self._type)
            #    lol = RR(len(self._type)+1)
            #    self._color = Color(self._type[0]/lol,self._type[1]/lol,self._type[2]/lol)                                            
        return self._color
            
    def sorted_type(self):
        return tuple(sorted(self._type))
        
    @cached_method
    def sign(self):
        r"""
        EXAMPLES::
        
            sage: from EkEkstar import kFace
            sage: kFace((0,0,0),(1,2,3,4,5)).sign()
            1
            sage: kFace((0,0,0),(1,2,3,4,4)).sign()
            0
            sage: kFace((0,0,0),(1,2,3,5,4)).sign()
            -1
        """
        sorted_type = self.sorted_type()
        if all(sorted_type[i] < sorted_type[i+1] for i in range(len(sorted_type)-1)):
            p = argsort(self._type) + 1
            return Permutation(p).sign()
        else:
            return 0

    def __repr__(self):
        r"""
        String representation.

        EXAMPLES::

            sage: from EkEkstar import kFace
            sage: kFace((0,0,0),(1,2,3,4,5))
            [(0, 0, 0), (1, 2, 3, 4, 5)]
            sage: kFace((0,0,0),(1,2,3,4,4))
            [(0, 0, 0), (1, 2, 3, 4, 4)]
            sage: kFace((0,0,0),(1,2,3,5,4))
            [(0, 0, 0), (1, 2, 3, 5, 4)]

        Dual face::

            sage: kFace((0,0,0), (1,2,3), d=1)
            [(0, 0, 0), (1, 2, 3)]*
        """
        dual = '*' if self.dual() == 1 else ''
        return "[{}, {}]{}".format(self.vector(), self.type(), dual)

    def __eq__(self, other):
        return (isinstance(other, kFace) and
                self.vector() == other.vector() and
                self.type() == other.type() and 
                self.dual() == other.dual())

    @cached_method
    def __hash__(self):
        return hash((self.vector(), self.type(), self.dual()))

    def __add__(self, other):
        r"""
        EXAMPLES::
        
            sage: from EkEkstar import kFace
            sage: kFace((0,0,0),(1,3)) + kFace((0,0,0),(3,1))
            Empty patch
            sage: kFace((0,0,0),(1,3)) + kFace((0,0,0),(1,3))
            Patch: 2[(0, 0, 0), (1, 3)]

        """
        if isinstance(other, kFace):
            return kPatch([self, other])
        else:
            return kPatch(other).union(self)

    def _plot(self, geosub):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace, GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: geosub = GeoSub(sub,2, 'prefix',1)
            sage: f = kFace((0,0,0),(1,2),d=1)
            sage: _ = f._plot(geosub)

        ::

            sage: sub = {1:[1,2,3,3,3,3], 2:[1,3], 3:[1]}
            sage: geosub = GeoSub(sub,2, 'prefix',1)
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
    r"""
    EXAMPLES::

        sage: from EkEkstar import kPatch, kFace
        sage: P = kPatch([kFace((0,0,0),(1,2),d=1),
        ....:             kFace((0,0,1),(1,3),d=1),
        ....:             kFace((0,1,0),(2,1),d=1),
        ....:             kFace((0,0,0),(3,1),d=1)])
        sage: P
        Patch: 1[(0, 0, 0), (1, 2)]* + 1[(0, 0, 1), (1, 3)]* + -1[(0, 1, 0), (1, 2)]* + -1[(0, 0, 0), (1, 3)]*
    """
    def __init__(self, faces, face_contour=None):
        r"""
        EXAMPLES::

        TESTS:

        Cancellation because of ordering of the type::

            sage: from EkEkstar import kFace, kPatch
            sage: L = [kFace((0,0,0),(1,3)), kFace((0,0,0),(3,1))]
            sage: kPatch(L)
            Empty patch

        Repetition in the list::

            sage: L = [kFace((0,0,0),(1,3)), kFace((0,0,0),(1,3))]
            sage: kPatch(L)
            Patch: 2[(0, 0, 0), (1, 3)]
        """
        # Compute the formal sum with support on canonical faces
        self._faces = Counter()
        if isinstance(faces, list):
            for f in faces:
                canonical = kFace(f.vector(), f.sorted_type(), d=f.dual(), color=f.color())
                self._faces[canonical] += f.sign()
        else:
            for (f,m) in faces.items():
                canonical = kFace(f.vector(), f.sorted_type(), d=f.dual(), color=f.color())
                self._faces[canonical] += m*f.sign()

        # Remove faces with multiplicty zero from the formal sum
        for f,m in self._faces.items():
            if m == 0:
                del self._faces[f]

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
                    1: map(vector, [(0,0,0),(0,1,0),(0,1,1),(0,0,1)]),
                    2: map(vector, [(0,0,0),(0,0,1),(1,0,1),(1,0,0)]),
                    3: map(vector, [(0,0,0),(1,0,0),(1,1,0),(0,1,0)])
            }
            
    def __len__(self):
        return len(self._faces)

    def __iter__(self):
        return iter(self._faces.items())
       
    def __add__(self, other):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace, kPatch
            sage: P = kPatch([kFace((0,0,0),(1,3))])
            sage: Q = kPatch([kFace((0,0,0),(3,1))])
            sage: P + Q
            Empty patch
            sage: Q + P
            Empty patch

        ::

            sage: P + P
            Patch: 2[(0, 0, 0), (1, 3)]

        ::

            sage: P = kPatch([kFace((0,0,0),(1,2),d=1),
            ....:             kFace((0,0,1),(1,3),d=1),
            ....:             kFace((0,1,0),(2,1),d=1)])
            sage: Q = kPatch([kFace((0,1,0),(2,1),d=1),
            ....:             kFace((0,0,0),(3,1),d=1)])
            sage: P + Q
            Patch: 1[(0, 0, 0), (1, 2)]* + -1[(0, 0, 0), (1, 3)]* + 1[(0, 0, 1), (1, 3)]* + -2[(0, 1, 0), (1, 2)]*
        """
        C = Counter()
        for (f,m) in self._faces.items():
            C[f] += m
        for (f,m) in other._faces.items():
            C[f] += m
        return kPatch(C)
        
    def dimension(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace, kPatch
            sage: P = kPatch([kFace((0,0,0),(1,2),d=1),
            ....:             kFace((0,0,1),(1,3),d=1),
            ....:             kFace((0,1,0),(2,1),d=1),
            ....:             kFace((0,0,0),(3,1),d=1)])
            sage: P.dimension()
            3
        """
        return self._dimension

    def __repr__(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace, kPatch
            sage: P = kPatch([kFace((0,0,0),(1,2),d=1),
            ....:             kFace((0,0,1),(1,3),d=1),
            ....:             kFace((0,1,0),(2,1),d=1),
            ....:             kFace((0,0,0),(3,1),d=1)])
            sage: P
            Patch: 1[(0, 0, 0), (1, 2)]* + 1[(0, 0, 1), (1, 3)]* + -1[(0, 1, 0), (1, 2)]* + -1[(0, 0, 0), (1, 3)]*

        With multiplicity::

            sage: P = kPatch({kFace((0,0,0),(1,2),d=1):11,
            ....:             kFace((0,0,1),(1,3),d=1):22,
            ....:             kFace((0,1,0),(2,1),d=1):33,
            ....:             kFace((0,0,0),(3,1),d=1):-44})
            sage: P
            Patch: 11[(0, 0, 0), (1, 2)]* + 44[(0, 0, 0), (1, 3)]* + 22[(0, 0, 1), (1, 3)]* + -33[(0, 1, 0), (1, 2)]*

        Empty patch::

            sage: kPatch([])
            Empty patch
        """
        if len(self) == 0:
            return "Empty patch"
        elif len(self) <= 30:
            L = ["{}{}".format(m,f) for (f,m) in sorted(self)]
            return "Patch: %s" % ' + '.join(L)
        else:
            return "Patch of %s faces"%len(self)
        
    def union(self, other):
        r"""
        INPUT:

        - ``other`` -- a face, a patch or a list of faces

        EXAMPLES::

            sage: from EkEkstar import kFace, kPatch
            sage: P = kPatch([kFace((0,0,0),(1,2),d=1),
            ....:             kFace((0,0,1),(1,3),d=1)])
            sage: f = kFace((0,1,0),(2,1),d=1)
            sage: g = kFace((0,0,0),(3,1),d=1)

        A patch union with a face::

            sage: P.union(f)
            Patch: 1[(0, 0, 0), (1, 2)]* + 1[(0, 0, 1), (1, 3)]* + -1[(0, 1, 0), (1, 2)]*

        A patch union with a patch::

            sage: P.union(P)
            Patch: 2[(0, 0, 0), (1, 2)]* + 2[(0, 0, 1), (1, 3)]*

        A patch union with a list of faces::

            sage: P.union([f,g])
            Patch: 1[(0, 0, 0), (1, 2)]* + -1[(0, 0, 0), (1, 3)]* + 1[(0, 0, 1), (1, 3)]* + -1[(0, 1, 0), (1, 2)]*

        """
        if isinstance(other, kFace):
            return self + kPatch([other])
        elif isinstance(other, kPatch):
            return self + other
        else:
            return self + kPatch(other)
        
    def plot(self, geosub):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace, kPatch, GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: geosub = GeoSub(sub,2, 'prefix',1)
            sage: P = kPatch([kFace((0,0,0),(1,2),d=1),
            ....:             kFace((0,0,1),(1,3),d=1),
            ....:             kFace((0,1,0),(2,1),d=1),
            ....:             kFace((0,0,0),(3,1),d=1)])
            sage: _ = P.plot(geosub)
        """
        G = Graphics()
        for face,m in self:
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
            
     
             

def ps_automaton(sub, presuf):
    r"""
    Return the prefix or suffix automaton

    (related to the prefix-suffix automaton).

    INPUT:

    - ``sub`` -- dict, 1 dimensional substitution
    - ``presuf`` -- string, ``"prefix"`` or ``"suffix"``

    OUTPUT:

        dict

    EXAMPLES::

        sage: from EkEkstar.EkEkstar import ps_automaton
        sage: m = {2:[2,1,1], 1:[2,1]}
        sage: ps_automaton(m, "prefix")
        {1: [(2, []), (1, [2])], 2: [(2, []), (1, [2]), (1, [2, 1])]}
        sage: ps_automaton(m, 'suffix')
        {1: [(2, [1]), (1, [])], 2: [(2, [1, 1]), (1, [1]), (1, [])]}

    """
    d = {}
    v = sub.values()
    for i in range(len(v)):
        L = []
        for j in range(len(v[i])):
            if presuf == "prefix":
                L.append((v[i][j],sub[i+1][0:j]))
            elif presuf == "suffix":
                L.append((v[i][j],sub[i+1][j+1:len(v[i])])) 
        d[i+1] = L  
    return d             
                  

def ps_automaton_inverted(sub, presuf):
    r"""
    Return the prefix or suffix automaton with inverted edges.

    (related to the prefix-suffix automaton).

    INPUT:

    - ``sub`` -- dict, 1 dimensional substitution
    - ``presuf`` -- string, ``"prefix"`` or ``"suffix"``

    OUTPUT:

        dict

    EXAMPLES::

        sage: from EkEkstar.EkEkstar import ps_automaton_inverted
        sage: m = {2:[2,1,1], 1:[2,1]}
        sage: ps_automaton_inverted(m, "prefix")
        {1: [(1, [2]), (2, [2]), (2, [2, 1])], 2: [(1, []), (2, [])]}
        sage: ps_automaton_inverted(m, 'suffix')
        {1: [(1, []), (2, [1]), (2, [])], 2: [(1, [1]), (2, [1, 1])]}

    """
    d = {}
    k = sub.keys()
    Gr = ps_automaton(sub, presuf)
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
    INPUT:

    - ``sigma`` -- dict, substitution
    - ``k`` -- integer
    - ``presuf`` -- string (default: ``"prefix"``), ``"prefix"`` or ``"suffix"`` 
    - ``dual`` -- integer (default: ``0``), 0 or 1

    EXAMPLES::

        sage: from EkEkstar import GeoSub
        sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
        sage: E = GeoSub(sub, 2)
        sage: E
        E_2(1->12, 2->13, 3->1)
    """
    def __init__(self, sigma, k, presuf='prefix', dual=0):
        self._sigma_dict = sigma
        self._sigma = WordMorphism(sigma)
        self._k = k
        if not presuf in ['prefix', 'suffix']:
            raise ValueError('Input presuf(={}) should be "prefix" or'
                    ' "suffix"'.format(presuf))
        self._presuf = presuf
        self._dual = dual

    @cached_method
    def field(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2)
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
            sage: E = GeoSub(sub,2)
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
            sage: E = GeoSub(sub,2)
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
            sage: E = GeoSub(sub,2)
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
            sage: E = GeoSub(sub,2)
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
            sage: E = GeoSub(sub,2)
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
            sage: E = GeoSub(sub,2)
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
            sage: E = GeoSub(sub,2)
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
                    bigL.append(ps_automaton(self._sigma_dict,self._presuf)[y])
                else:
                    bigL.append(ps_automaton_inverted(self._sigma_dict,self._presuf)[y])
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
            sage: geosub = GeoSub(sub,2, 'prefix',1)
            sage: P = kPatch([kFace((0,0,0),(1,2),d=1),
            ....:             kFace((0,0,1),(1,3),d=1),
            ....:             kFace((0,1,0),(2,1),d=1),
            ....:             kFace((0,0,0),(3,1),d=1)])
            sage: Q = geosub(P, 6)
            sage: Q
            Patch of 47 faces
        """
        if iterations == 0:
            return kPatch(patch)
        elif iterations < 0:
            raise ValueError("iterations (=%s) must be >= 0." % iterations)
        else:
            old_faces = patch
            for i in range(iterations):
                new_faces = kPatch([])
                for f,m in old_faces:
                    if m != 0:
                        new_faces += kPatch(self._call_on_face(f, color=f.color()))
                old_faces = new_faces
            return new_faces

    def matrix(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2)
            sage: E.matrix()
            [1 1 1]
            [1 0 0]
            [0 1 0]
        """
        if self._dual == 0:
            return self._sigma.incidence_matrix()
        else:
            return self._sigma.incidence_matrix().inverse()
        
    def _call_on_face(self, face, color=None):
        r"""
        INPUT:

        - ``face`` -- a face
        - ``color`` -- a color or None

        OUTPUT:

            dict of the form ``{face:multiplicity for face in faces}``

        EXAMPLES::
            
            sage: from EkEkstar import GeoSub, kFace
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2)

        The available face type are::

            sage: E.base_iter().keys()
            [(1, 2), (1, 3), (2, 3)]

        For each we get::

            sage: d = E._call_on_face(kFace((10,11,12), (1,2)))
            sage: sorted(d.items())
            [([(34, 10, 11), (1, 3)], 1),
             ([(33, 10, 11), (1, 1)], 1),
             ([(34, 10, 11), (2, 1)], 1),
             ([(35, 10, 11), (2, 3)], 1)]
            sage: sorted(E._call_on_face(kFace((10,11,12), (1,3))).items())
            [([(34, 10, 11), (2, 1)], 1), ([(33, 10, 11), (1, 1)], 1)]
            sage: E._call_on_face(kFace((10,11,12), (2,3)))
            {[(33, 10, 11), (1, 1)]: 1, [(34, 10, 11), (3, 1)]: 1}

        TESTS:

        This is an error::

            sage: E._call_on_face(kFace((10,11,12), (2,4)))
            Traceback (most recent call last):
            ...
            KeyError: (2, 4)
        """
        x_new = self.matrix() * face.vector()
        t = face.type()
        D = self._dual
        if D == 1:
            return {kFace(x_new - vv, tt, d=D):(-1)**(sum(t)+sum(tt))*face.sign()
                    for (vv, tt) in self.base_iter()[t] if len(tt) == self._k}
        else:
            return {kFace(x_new + vv, tt, d=D):face.sign()
                    for (vv, tt) in self.base_iter()[t] if len(tt) == self._k}
                
    def __repr__(self):
        r"""
        EXAMPLES::
            
            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: GeoSub(sub, 2,  'prefix', 0)
            E_2(1->12, 2->13, 3->1)
            sage: GeoSub(sub, 2,  'prefix', 1)
            E*_2(1->12, 2->13, 3->1)
        """
        if self._dual == 0: 
            return "E_%s(%s)" % (self._k,str(self._sigma))
        else: 
            return "E*_%s(%s)" % (self._k,str(self._sigma))
        


