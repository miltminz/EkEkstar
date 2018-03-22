==============================================================================
Demo of EkEkstar -- Milton Minervino -- LaBRI Sage Thursdays -- March 22, 2018
==============================================================================

Notes taken by Sébastien Labbé

k-Faces::

    sage: from EkEkstar import kFace,kPatch,GeoSub
    sage: F1 = kFace((0,0,0),(1,2))
    sage: F1
    [(0, 0, 0), (1, 2)]
    sage: F1.vector()
    (0, 0, 0)
    sage: F1.type()
    (1, 2)

k-Patches are formal sums of k-faces::

    sage: F3 = kFace((0,0,0,6,7),(1,4,5,2))
    sage: F3
    [(0, 0, 0, 6, 7), (1, 4, 5, 2)]
    sage: F3 + F1
    Patch: -1[(0, 0, 0), (1, 2)] + 1[(0, 0, 0, 6, 7), (1, 2, 4, 6)]

Adding k-faces may cancel depending on their multiplicity and the permutation
sign of their type::

    sage: F2 = kFace((0,0,0),(2,1))
    sage: F2
    [(0, 0, 0), (2, 1)]
    sage: F1 + F2
    Empty patch

Creation of k-Patches::

    sage: F1
    [(0, 0, 0), (1, 2)]
    sage: kPatch([F1,F1])
    Patch: 2[(0, 0, 0), (1, 2)]
    sage: F1 + F1
    Patch: 2[(0, 0, 0), (1, 2)]
    sage: F2
    [(0, 0, 0), (2, 1)]
    sage: kPatch([F2])
    Patch: -1[(0, 0, 0), (1, 2)]

BUG::

    sage: kFace((0,0,0),(1,2)) + kFace((0,0,1),(3,1)) + kFace((13,23,34),(1,1))
    Traceback (most recent call last)
    ...
    AttributeError: 'kFace' object has no attribute '_faces'

Dual k-faces::

    sage: F = kFace((0,0,0,0),(1,3,2))
    sage: F+F
    Patch: -2[(0, 0, 0, 0), (1, 2, 3)]
    sage: F._dual
    False
    sage: Fd = F.dual()
    sage: Fd = kFace((0,0,0,0),(1,3,2),dual=True)
    sage: Fd.is_dual()
    True
    sage: Fd
    [(0, 0, 0, 0), (1, 3, 2)]*

The EkEkStar module generalizes what is already in Sage::

    sage: from sage.combinat.e_one_star import Face, Patch, E1Star

Substitutions are already in Sage::

    sage: sub = {1:[1,2], 2:[1,3], 3:[1]}
    sage: sub[1]
    [1, 2]
    sage: s = WordMorphism(sub)
    sage: s
    WordMorphism: 1->12, 2->13, 3->1
    sage: s(1)
    word: 12
    sage: s([2])
    word: 13
    sage: s([1,1,1,1])
    word: 12121212
    sage: s([2,1,3])
    word: 13121

Iteration of the Tribonacci substitution on letter 1::

    sage: s(1,1)
    word: 12
    sage: s(1,2)
    word: 1213
    sage: s(1,3)
    word: 1213121
    sage: s(1,4)
    word: 1213121121312

The word 121312112131212131211213 defines a path in $\ZZ^3$. This can be
formalized by a geometric substitution (``GeoSub``). This is useless for
3d-path coded by words since concatenation of words `uv` is very easy and the
geometric position of the start of the word `v` is simply the end position of
the word `u` seen as a path. But this concatenation rule works only for words
and it can't help for applying a substitution on a set of faces. Therefore, we
need to consider k-Patches of faces as formal sums.

Here we consider the GeoSub in the easy case of faces of dimension 1 which can
be seen just as concatenation of paths::

    sage: E1 = GeoSub(sub,1,dual=False)
    sage: E1
    E_1(1->12, 2->13, 3->1)
    sage: P = kPatch([kFace((0,0,0),(1,))])
    sage: P
    Patch: 1[(0, 0, 0), (1,)]
    sage: E1(P)
    Patch: 1[(1, 0, 0), (2,)] + 1[(0, 0, 0), (1,)]
    sage: E1(P,2)
    Patch: 1[(1, 1, 0), (1,)] + 1[(2, 1, 0), (3,)] + 1[(0, 0, 0), (1,)] + 1[(1, 0, 0), (2,)]
    sage: E1(P,7)
    Patch of 81 faces
    sage: E1(P,5).plot(E1)
    Launched png viewer for Graphics object consisting of 24 graphics primitives

Now we consider dual faces which are not segments anymore and for which the
formal sums formalism is necessary (this corresponds to the E1star that is
already in Sage)::

    sage: E1star = GeoSub(sub,1,dual=True)
    sage: E1star
    E*_1(1->12, 2->13, 3->1)
    sage: P
    Patch: 1[(0, 0, 0), (1,)]
    sage: Pstar = P.dual()
    sage: Pstar = kPatch([kFace((0,0,0),(1,),dual=True)])
    sage: Pstar
    Patch: 1[(0, 0, 0), (1,)]*
    sage: E1(P)
    Patch: 1[(1, 0, 0), (2,)] + 1[(0, 0, 0), (1,)]
    sage: E1star(Pstar)
    Patch: 1[(0, 0, 0), (1,)]* + -1[(0, 0, 0), (2,)]* + 1[(0, 0, 0), (3,)]*

In the EkEkstar module, the projection is done in the contracting plane::

    sage: E1star(Pstar).plot(E1star)
    Launched png viewer for Graphics object consisting of 3 graphics primitives
    sage: E1star(Pstar,5).plot(E1star)
    Launched png viewer for Graphics object consisting of 31 graphics primitives
    sage: E1star(Pstar,7).plot(E1star)
    Launched png viewer for Graphics object consisting of 105 graphics primitives

The module allows more general geometric substitution like $E_2^*$ which
computes the boundary of the fractal::

    sage: E2star = GeoSub(sub,2,dual=True)
    sage: E2star
    E*_2(1->12, 2->13, 3->1)
    sage: E2star.base_iter()
    {(1, 2): [[(0, 0, 0), (1,)],
    [(0, 0, 0), (2,)],
    [(0, 0, 0), (3,)],
    [(0, 0, -1), (1, 1)],
    [(0, 0, -1), (2, 1)],
    [(0, 0, -1), (3, 1)]],
    (1, 3): [[(0, 0, 0), (1,)],
    [(0, 0, 0), (2,)],
    [(0, 0, 0), (3,)],
    [(0, 0, -1), (1, 2)],
    [(0, 0, -1), (2, 2)],
    [(0, 0, -1), (3, 2)]],
    (2, 3): [[(0, 0, -1), (1,)], [(0, 0, -2), (1, 2)]]}
    sage: P = kPatch([kFace((0,0,0),(1,2),dual=True)])
    sage: E2star(P,6).plot(E2star)
    Launched png viewer for Graphics object consisting of 13 graphics primitives

