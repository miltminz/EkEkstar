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

    - Patch.dual() and Face.dual() should return the dual face

    - Face + Face + Face returns an error

    - Patch should check that all its faces are consistent (same type
      length, dual or not, ambiant dimension)

    - Should be able to do 4 * kPatch

EXAMPLES::

    sage: from EkEkstar import GeoSub, kPatch, kFace
    sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
    sage: geosub = GeoSub(sub,2, presuf='prefix', dual=True)
    sage: P = kPatch([kFace((0,0,0),(1,2),dual=True),
    ....:             kFace((0,0,1),(1,3),dual=True),
    ....:             kFace((0,1,0),(2,1),dual=True),
    ....:             kFace((0,0,0),(3,1),dual=True)])
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

    INPUT:

    - ``v`` -- vector
    - ``t`` -- tuple, type
    - ``dual`` -- bool (default:``False``)
    - ``color`` -- string (default:``None``)

    EXAMPLES:

    Face based at (0,0,0) of type (1,2)::

        sage: from EkEkstar import kFace
        sage: F = kFace((0,0,0),(1,2))
        sage: F
        [(0, 0, 0), (1, 2)]
        
    Face based at (0,0,0) of type (3,1)::

        sage: kFace((0,0,0),(3,1))
        [(0, 0, 0), (3, 1)]
        
    Dual face based at (0,0,0,0) of type (1)::
        
        sage: kFace((0,0,0,0),(1), dual=True)
        [(0, 0, 0, 0), 1]*

    Operations::

        sage: F = kFace((0,0,0),(1,2))
        sage: F
        [(0, 0, 0), (1, 2)]
        sage: -2 * F.dual()
        Patch: -2[(0, 0, 0), (1, 2)]*
        
    Color of a face::
    
        sage: F = kFace((0,0,0),(1,2))
        sage: F.color()
        RGB color (1.0, 0.0, 0.0)

        sage: F = kFace((0,0,0),(1,2),color='green')
        sage: F.color()
        RGB color (0.0, 0.5019607843137255, 0.0)
        
    """
    def __init__(self, v, t, dual=False, color=None):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace
            sage: F = kFace((0,0,0),(1,2))
            sage: F
            [(0, 0, 0), (1, 2)]
        """
        self._vector = (ZZ**len(v))(v)
        self._vector.set_immutable()
        self._dimension = len(v)
        
        if not all((tt in ZZ and 1 <= tt <= len(v)) for tt in t):
            raise ValueError('The type must be a tuple of integers between 1 and {}'.format(len(v)))

        self._type = t
        
        self._dual = dual

        self._color = color
        
        if color is not None:
        
            self._color = Color(color)
            
        else:
            
            sorted_types = list(combinations(range(1,self._dimension+1),len(self._type)))
            Col = rainbow(len(sorted_types))
            D = dict(zip(sorted_types,Col))
        
            self._color = Color(D[self.sorted_type()])



    def vector(self):
        return self._vector

    def type(self):
        return self._type

    def is_dual(self):
        return self._dual


    def sorted_type(self):
        return tuple(sorted(self._type))


    def color(self):
        return self._color
        
        
            
        
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

            sage: kFace((0,0,0), (1,2,3), dual=True)
            [(0, 0, 0), (1, 2, 3)]*
        """
        d = '*' if self.is_dual() else ''
        return "[{}, {}]{}".format(self.vector(), self.type(), d)

    def __eq__(self, other):
        return (isinstance(other, kFace) and
                self.vector() == other.vector() and
                self.type() == other.type() and 
                self.is_dual() == other.is_dual())

    @cached_method
    def __hash__(self):
        return hash((self.vector(), self.type(), self.is_dual()))

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

    def __neg__(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kPatch
            sage: F = kFace((0,0,0),(1,3))
            sage: F
            [(0, 0, 0), (1, 3)]
            sage: -F
            Patch: -1[(0, 0, 0), (1, 3)]

        """
        return kPatch({self:-1})

    def __rmul__(self, coeff):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kPatch
            sage: F = kFace((0,0,0),(1,3))
            sage: F
            [(0, 0, 0), (1, 3)]
            sage: 4 * F
            Patch: 4[(0, 0, 0), (1, 3)]
            sage: -2 * F
            Patch: -2[(0, 0, 0), (1, 3)]

        """
        return kPatch({self:coeff})

    def dual(self):
        r"""
        Return the dual face.

        EXAMPLES::

            sage: from EkEkstar import kFace
            sage: kFace((0,0,0),(1,3))
            [(0, 0, 0), (1, 3)]
            sage: kFace((0,0,0),(1,3)).dual()
            [(0, 0, 0), (1, 3)]*
            sage: kFace((0,0,0),(1,3)).dual().dual()
            [(0, 0, 0), (1, 3)]

        """
        return kFace(self.vector(), self.type(), dual=not self.is_dual(), 
                     color=self.color())

    def _plot(self, geosub, color=None):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace, GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: geosub = GeoSub(sub,2, dual=True)
            sage: f = kFace((0,0,0),(1,2), dual=True)
            sage: _ = f._plot(geosub)

        ::

            sage: sub = {1:[1,2,3,3,3,3], 2:[1,3], 3:[1]}
            sage: geosub = GeoSub(sub,2, dual=True)
            sage: f = kFace((0,0,0),(1,2), dual=True)
            sage: _ = f._plot(geosub)
        """
        v = self.vector()
        t = self.type()
        if color != None:
            col = color
        else:
            col = self._color
        G = Graphics()
        
        K = geosub.field()
        b = K.gen()
                
        num = geosub._sigma_dict.keys()
        
        if self.is_dual():
            h = list(set(num)-set(t))
            B = b
            vec = geosub.dominant_left_eigenvector()
            emb = geosub.contracting_eigenvalues_indices()
        else:
            h = list(t)
            B = b**(-1)  # TODO: this seems useless (why?)
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
        sage: P = kPatch([kFace((0,0,0),(1,2),dual=True),
        ....:             kFace((0,0,1),(1,3),dual=True),
        ....:             kFace((0,1,0),(2,1),dual=True),
        ....:             kFace((0,0,0),(3,1),dual=True)])
        sage: P
        Patch: 1[(0, 0, 0), (1, 2)]* + 1[(0, 0, 1), (1, 3)]* + -1[(0, 1, 0), (1, 2)]* + -1[(0, 0, 0), (1, 3)]*
    """
    def __init__(self, faces):
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
                canonical = kFace(f.vector(), f.sorted_type(), dual=f.is_dual(), color=f.color())
                self._faces[canonical] += f.sign()
        else:
            for (f,m) in faces.items():
                canonical = kFace(f.vector(), f.sorted_type(), dual=f.is_dual(), color=f.color())
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

            sage: P = kPatch([kFace((0,0,0),(1,2),dual=True),
            ....:             kFace((0,0,1),(1,3),dual=True),
            ....:             kFace((0,1,0),(2,1),dual=True)])
            sage: Q = kPatch([kFace((0,1,0),(2,1),dual=True),
            ....:             kFace((0,0,0),(3,1),dual=True)])
            sage: P + Q
            Patch: 1[(0, 0, 0), (1, 2)]* + -1[(0, 0, 0), (1, 3)]* + 1[(0, 0, 1), (1, 3)]* + -2[(0, 1, 0), (1, 2)]*
        """
        C = Counter()
        for (f,m) in self._faces.items():
            C[f] += m
        for (f,m) in other._faces.items():
            C[f] += m
        return kPatch(C)

    def __rmul__(self, coeff):
        r"""
        INPUT:

        - ``coeff`` -- integer

        EXAMPLES::

            sage: from EkEkstar import kPatch
            sage: P = kPatch([kFace((0,0,0),(1,3))])
            sage: P
            Patch: 1[(0, 0, 0), (1, 3)]

        ::

            sage: -4 * P
            Patch: -4[(0, 0, 0), (1, 3)]
            sage: -1 * P
            Patch: -1[(0, 0, 0), (1, 3)]
            sage: 1 * P
            Patch: 1[(0, 0, 0), (1, 3)]

        Currently, it is not forbidden to use non integral coefficients::

            sage: -4.3 * P
            Patch: -4.30000000000000[(0, 0, 0), (1, 3)]

        TESTS:

        Right multiplication is not defined::

            sage: P * 2
            Traceback (most recent call last):
            ...
            TypeError: unsupported operand parent(s) for *: '<class '__main__.kPatch'>' and 'Integer Ring'

        """
        D = {f:coeff*m for (f,m) in self._faces.items()}
        return kPatch(D)

    def __neg__(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kPatch
            sage: P = kPatch([kFace((0,0,0),(1,3))])
            sage: P
            Patch: 1[(0, 0, 0), (1, 3)]
            sage: -P
            Patch: -1[(0, 0, 0), (1, 3)]

        """
        D = {f:-m for (f,m) in self._faces.items()}
        return kPatch(D)

    def dual(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kPatch
            sage: P = kPatch([kFace((0,1,0),(1,2)), kFace((0,0,0),(1,3))])
            sage: P
            Patch: 1[(0, 1, 0), (1, 2)] + 1[(0, 0, 0), (1, 3)]
            sage: P.dual()
            Patch: 1[(0, 1, 0), (1, 2)]* + 1[(0, 0, 0), (1, 3)]*

        """
        D = {f.dual():m for (f,m) in self._faces.items()}
        return kPatch(D)

    def dimension(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace, kPatch
            sage: P = kPatch([kFace((0,0,0),(1,2),dual=True),
            ....:             kFace((0,0,1),(1,3),dual=True),
            ....:             kFace((0,1,0),(2,1),dual=True),
            ....:             kFace((0,0,0),(3,1),dual=True)])
            sage: P.dimension()
            3
        """
        return self._dimension

    def __repr__(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace, kPatch
            sage: P = kPatch([kFace((0,0,0),(1,2),dual=True),
            ....:             kFace((0,0,1),(1,3),dual=True),
            ....:             kFace((0,1,0),(2,1),dual=True),
            ....:             kFace((0,0,0),(3,1),dual=True)])
            sage: P
            Patch: 1[(0, 0, 0), (1, 2)]* + 1[(0, 0, 1), (1, 3)]* + -1[(0, 1, 0), (1, 2)]* + -1[(0, 0, 0), (1, 3)]*

        With multiplicity::

            sage: P = kPatch({kFace((0,0,0),(1,2),dual=True):11,
            ....:             kFace((0,0,1),(1,3),dual=True):22,
            ....:             kFace((0,1,0),(2,1),dual=True):33,
            ....:             kFace((0,0,0),(3,1),dual=True):-44})
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
            sage: P = kPatch([kFace((0,0,0),(1,2),dual=True),
            ....:             kFace((0,0,1),(1,3),dual=True)])
            sage: f = kFace((0,1,0),(2,1),dual=True)
            sage: g = kFace((0,0,0),(3,1),dual=True)

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
        
    def plot(self, geosub, color=None):
        r"""
        EXAMPLES::

            sage: from EkEkstar import kFace, kPatch, GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: geosub = GeoSub(sub,2, dual=True)
            sage: P = kPatch([kFace((0,0,0),(1,2),dual=True),
            ....:             kFace((0,0,1),(1,3),dual=True),
            ....:             kFace((0,1,0),(2,1),dual=True),
            ....:             kFace((0,0,0),(3,1),dual=True)])
            sage: _ = P.plot(geosub)
        """
        G = Graphics()
        for face,m in self:
            G += face._plot(geosub,color)
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
    - ``dual`` -- bool (default: ``False``)

    EXAMPLES::

        sage: from EkEkstar import GeoSub
        sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
        sage: E = GeoSub(sub, 2)
        sage: E
        E_2(1->12, 2->13, 3->1)
    """
    def __init__(self, sigma, k, presuf='prefix', dual=False):
        self._sigma_dict = sigma
        self._sigma = WordMorphism(sigma)
        self._k = k
        if not presuf in ['prefix', 'suffix']:
            raise ValueError('Input presuf(={}) should be "prefix" or'
                    ' "suffix"'.format(presuf))
        self._presuf = presuf
        self._dual = dual

    def is_dual(self):
        return self._dual

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
        b1 = max(M.eigenvalues(), key=abs)
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
        if self.is_dual():
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

    def minkowski_embedding_with_left_eigenvector(self):
        r"""
        EXAMPLES::

            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: E = GeoSub(sub,2)
            sage: E.minkowski_embedding_with_left_eigenvector()
            [ -1.00000000000000 -0.839286755214161 -0.543689012692076]
            [ -1.00000000000000   1.41964337760708  0.771844506346038]
            [ 0.000000000000000 -0.606290729207199   1.11514250803994]

        ::

            sage: E = GeoSub(sub, 2, dual=True)
            sage: E.minkowski_embedding_with_left_eigenvector()
            [  1.00000000000000  0.839286755214161  0.543689012692076]
            [  1.00000000000000  -1.41964337760708 -0.771844506346038]
            [ 0.000000000000000  0.606290729207199  -1.11514250803994]

        """
        K = self.field()
        if self.is_dual():
            vb = self.dominant_left_eigenvector()
        else:
            vb = -self.dominant_left_eigenvector() 
        return Minkowski_embedding_without_sqrt2(K, vb)

    def contracting_rauzy_projection(self):
        raise NotImplementedError
    def expanding_rauzy_projection(self):
        raise NotImplementedError

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
                if self.is_dual():
                    bigL.append(ps_automaton_inverted(self._sigma_dict,self._presuf)[y])
                else:
                    bigL.append(ps_automaton(self._sigma_dict,self._presuf)[y])
                Lpro = list(product(*bigL))
                for el in Lpro:
                    z = []
                    w = []
                    for i in range(len(el)):
                        z += el[i][1]
                        w.append(el[i][0])
                    if self.is_dual():
                        M = self._sigma.incidence_matrix()
                        X[x].append([-M.inverse()*abelian(z, S),tuple(w)])
                    else:
                        X[x].append([abelian(z, S),tuple(w)])
        return X
           
    def __call__(self, patch, iterations=1):
        r"""
        EXAMPLES::

            sage: from EkEkstar import GeoSub, kPatch, kFace
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: geosub = GeoSub(sub,2, 'prefix',1)
            sage: P = kPatch([kFace((0,0,0),(1,2),dual=True),
            ....:             kFace((0,0,1),(1,3),dual=True),
            ....:             kFace((0,1,0),(2,1),dual=True),
            ....:             kFace((0,0,0),(3,1),dual=True)])
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
                        new_faces += kPatch(self._call_on_face(f,m, color=f.color()))              
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
        if self.is_dual():
            return self._sigma.incidence_matrix().inverse()
        else:
            return self._sigma.incidence_matrix()
        
    def _call_on_face(self, face, m=1, color=None):
        r"""
        INPUT:

        - ``face`` -- a face
        - ``m`` -- multiplicity
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
        #t = face.type()
        t = face.sorted_type()
        s = m*face.sign()
        if self.is_dual():
            return {kFace(x_new - vv, tt, dual=self.is_dual()):(-1)**(sum(t)+sum(tt))*s
                    for (vv, tt) in self.base_iter()[t] if len(tt) == self._k}
        else:
            return {kFace(x_new + vv, tt, dual=self.is_dual()):s
                    for (vv, tt) in self.base_iter()[t] if len(tt) == self._k}
                
    def __repr__(self):
        r"""
        EXAMPLES::
            
            sage: from EkEkstar import GeoSub
            sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
            sage: GeoSub(sub, 2)
            E_2(1->12, 2->13, 3->1)
            sage: GeoSub(sub, 2, dual=True)
            E*_2(1->12, 2->13, 3->1)
        """
        if self.is_dual(): 
            return "E*_%s(%s)" % (self._k,str(self._sigma))
        else: 
            return "E_%s(%s)" % (self._k,str(self._sigma))
        

def Minkowski_embedding_without_sqrt2(self, B=None, prec=None):
    r"""
    This method is a modification of the ``Minkowski_embedding`` method of
    NumberField in sage (without sqrt2).

    EXAMPLES::

        sage: from EkEkstar.EkEkstar import Minkowski_embedding_without_sqrt2
        sage: F.<alpha> = NumberField(x^3+2)
        sage: F.minkowski_embedding()
        [ 1.00000000000000 -1.25992104989487  1.58740105196820]
        [ 1.41421356237309 0.890898718140339 -1.12246204830937]
        [0.000000000000000  1.54308184421705  1.94416129723967]
        sage: Minkowski_embedding_without_sqrt2(F)
        [  1.00000000000000  -1.25992104989487   1.58740105196820]
        [  1.00000000000000  0.629960524947437 -0.793700525984099]
        [ 0.000000000000000   1.09112363597172   1.37472963699860]
        sage: Minkowski_embedding_without_sqrt2(F, [1, alpha+2, alpha^2-alpha])
        [ 1.00000000000000 0.740078950105127  2.84732210186307]
        [ 1.00000000000000  2.62996052494744 -1.42366105093154]
        [0.000000000000000  1.09112363597172 0.283606001026881]
        sage: Minkowski_embedding_without_sqrt2(F) * (alpha + 2).vector().column()
        [0.740078950105127]
        [ 2.62996052494744]
        [ 1.09112363597172]

    Tribo::

        sage: F.<beta> = NumberField(x^3-x^2-x-1)
        sage: F.minkowski_embedding()
        [  1.00000000000000   1.83928675521416   3.38297576790624]
        [  1.41421356237309 -0.593465355971987 -0.270804762516626]
        [ 0.000000000000000  0.857424571985895 -0.719625086862932]
        sage: Minkowski_embedding_without_sqrt2(F)
        [  1.00000000000000   1.83928675521416   3.38297576790624]
        [  1.00000000000000 -0.419643377607080 -0.191487883953119]
        [ 0.000000000000000  0.606290729207199 -0.508851778832738]

    Comprendre le problème de norme::

        sage: norme = lambda v:abs(v[0]) * (v[1]^2 + v[2]^2)
        sage: F.<beta> = NumberField(x^3-x^2-x-1)
        sage: M = Minkowski_embedding_without_sqrt2(F)
        sage: norme(M*vector((1,0,0)))
        1.00000000000000
        sage: norme(M*vector((1,0,-1)))
        4.00000000000000

    """
    r,s = self.signature()
    places = self.places(prec=prec)

    if B is None:
        B = [(self.gen(0))**i for i in range(self.degree())]

    d = {}
    for col,B_col in enumerate(B):
        for row in range(r):
            d[(row,col)] = places[row](B_col)
        for i in range(s):
            z = places[r+i](B_col)
            d[(r+2*i,col)] = z.real()
            d[(r+2*i+1,col)] = z.imag()

    from sage.matrix.constructor import matrix
    return matrix(d)

