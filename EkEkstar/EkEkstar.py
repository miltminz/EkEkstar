r"""
EkEkStar

AUTHOR:

 - Milton Minervino, 2017
 

EXAMPLES::

    sage: # global examples

This is a remainder for me::

    sage: import_statements('Graphics')
    from sage.plot.graphics import Graphics

"""
from itertools import product
from sage.structure.sage_object import SageObject
from sage.rings.integer_ring import ZZ
from sage.modules.free_module_element import vector
from sage.groups.perm_gps.permgroup_named import SymmetricGroup
from sage.rings.all import CC
from sage.combinat.words.morphism import WordMorphism
from sage.rings.number_field.number_field import NumberField
from sage.plot.colors import Color
from sage.plot.graphics import Graphics
from sage.plot.polygon import polygon2d






#  GLOBAL FUNCTIONS
def numfield(sub):
    M = WordMorphism(sub).incidence_matrix()
    b1 = max(M.eigenvalues())
    f = b1.minpoly()
    K = NumberField(f, 'b')
    b, = K.gens()
    return K  
    

def le(sub):
    return (WordMorphism(sub).incidence_matrix()-numfield(sub).gen()*1).kernel().basis()[0]
    #return vector([lefteig[0],lefteig[0]-lefteig[1]+lefteig[2],lefteig[1],-lefteig[1]+lefteig[3]])
    #return vector([lefteig[0]+lefteig[1],-lefteig[1]+lefteig[2]-lefteig[3],lefteig[1],lefteig[3]])
    
    

def re(sub):
    return (WordMorphism(sub).incidence_matrix().transpose()-numfield(sub).gen()*1).kernel().basis()[0]    
    

def embb(sub):
    return numfield(sub).gen().complex_embeddings()
    

def contr(sub):
    return [embb(sub).index(x) for x in embb(sub) if abs(x)<1]
    

def dil(sub):
    return [embb(sub).index(x) for x in embb(sub) if abs(x)>1]
    
        

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
                self.dual() == other.dual()
               )
    
    
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
            
    def _plot(self):
        
        v = self.vector()
        t = self.type()
        col = self._color
        G = Graphics()
        
        K = numfield(sub)
        b = K.gen()
                
        num = sub.keys()
        
        if self._dual == 1:
            h = list(set(num)-set(t))
            B = b
            vec = le(sub)
            emb = contr(sub)
        else:
            h = list(t)
            B = b^(-1) 
            vec = -le(sub) 
            emb = dil(sub) 
        
        el = v*vec
        iter = 0
        
        if len(h) == 1:
            if embb(sub)[emb[0]].is_real() == True:
                bp = zero_vector(CC, len(emb))
                for i in range(len(emb)):
                    bp[i] = K(B^(iter)*el).complex_embeddings()[emb[i]]
                bp1 = zero_vector(CC, len(emb))
                for i in range(len(emb)):
                    bp1[i] = K(B^(iter)*(el+vec[h[0]-1])).complex_embeddings()[emb[i]] 
                if len(emb) == 1:
                    return line([bp[0],bp1[0]],color = col,thickness = 3) 
                else: return line([bp,bp1],color = col,thickness = 3)      
            else: 
                bp = K(B^(iter)*el).complex_embeddings()[emb[0]]
                bp1 = K(B^(iter)*(el+vec[h[0]-1])).complex_embeddings()[emb[0]]
                return line([bp,bp1],color = col,thickness = 3)
        elif len(h) == 2: 
            if embb(sub)[emb[0]].is_real() == True:
                bp = ( K(B^(iter)*el).complex_embeddings()[emb[0]], K(B^(iter)*el).complex_embeddings()[emb[1]])
                bp1 = ( K(B^(iter)*(el+vec[h[0]-1])).complex_embeddings()[emb[0]], K(B^(iter)*(el+vec[h[0]-1])).complex_embeddings()[emb[1]] )
                bp2 = ( K(B^(iter)*(el+vec[h[0]-1]+vec[h[1]-1])).complex_embeddings()[emb[0]], K(B^(iter)*(el+vec[h[0]-1]+vec[h[1]-1])).complex_embeddings()[emb[1]])
                bp3 = ( K(B^(iter)*(el+vec[h[1]-1])).complex_embeddings()[emb[0]], K(B^(iter)*(el+vec[h[1]-1])).complex_embeddings()[emb[1]])
                return polygon2d([bp,bp1,bp2,bp3],color=col,thickness=.1,alpha = 0.8)
            else:   
                bp = K(B^(iter)*el).complex_embeddings()[emb[0]]
                bp1 = K(B^(iter)*(el+vec[h[0]-1])).complex_embeddings()[emb[0]]
                bp2 = K(B^(iter)*(el+vec[h[0]-1]+vec[h[1]-1])).complex_embeddings()[emb[0]]
                bp3 = K(B^(iter)*(el+vec[h[1]-1])).complex_embeddings()[emb[0]]
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
        
    
    def plot(self):
        
        G = Graphics()
        for face in self:
            G += face._plot()
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
            
     
             

def psauto(sub,ps):
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
      

def abel(L):
    return vector([L.count(i) for i in sub.keys()])

def GeoSub(patch,sub,presuf,dual):
    Lwed = []
    M = s.incidence_matrix()
    for x in patch:
        bigL= []
        if dual == 0:
            for y in x.type():
                bigL.append(psauto(sub,presuf)[y])
        else:
            for y in x.type():
                bigL.append(invauto(sub,presuf)[y])
        Lpro = list(product(*bigL))
        for el in Lpro:
            z = []
            w = []
            for i in range(len(el)):
                z += el[i][1]
                w.append(el[i][0]) 
            if dual == 0:
                Lwed.append(kFace(M*x.vector() + abel(z),tuple(w),x.mult(),dual))
            else:
                Lwed.append(kFace(M.inverse()*(x.vector() - abel(z)),tuple(w),x.mult(),dual))
    return kPatch(Lwed) 

class geosub(SageObject):
    
    def __init__(self, sigma, k, presuf, dual):
        
        self._sigma = WordMorphism(sigma)
        M = self._sigma.incidence_matrix()
        b1 = max(M.eigenvalues())
        f = b1.minpoly()
        K = NumberField(f, 'b')
        b, = K.gens()
        #K.<b> = NumberField(M.characteristic_polynomial())
        #self._matrix() = M
        
        self._d = len(sigma.keys())
        self._k = k
        self._dual = dual
        self._field = K        
        S = sigma.keys() 
        alph = [x for x in tuples(S,k) if all(x[i] < x[i+1] for i in range(k-1))]
        
        X = {}
    
        for x in alph:
            X[x] = []
            bigL = []
            for y in x:
                if self._dual == 0:
                    bigL.append(psauto(sigma,presuf)[y])
                else:
                    bigL.append(invauto(sigma,presuf)[y])
                Lpro = list(product(*bigL))
                for el in Lpro:
                    z = []
                    w = []
                    for i in range(len(el)):
                        z += el[i][1]
                        w.append(el[i][0])
                    if self._dual == 0:
                        X[x].append([abel(z),tuple(w)])
                    else:
                        X[x].append([-M.inverse()*abel(z),tuple(w)])
        
        self._base_iter = X
        
            
           
    def __call__(self, patch, iterations=1):
    
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
            return (kFace(x_new + (-1)^(self._dual)*y1, y2, (-1)^(sum(t)+sum(y2))*face.mult(), D) for (y1, y2) in self._base_iter[t] if len(y2) == self._k)
        else:
            return (kFace(x_new + (-1)^(self._dual)*y1, y2, face.mult(), D) for (y1, y2) in self._base_iter[t] if len(y2) == self._k)
                
    
    def __repr__(self):
        
        if self._dual==0: return "E_%s(%s)" % (self._k,str(self._sigma))
        else: return "E*_%s(%s)" % (self._k,str(self._sigma))
        
    def field(self):
        
        return self._field
        
    def gen(self):
        
        if self._dual == 1:
            return self._field.gen()
        else:
            return (self._field.gen())^(-1)
            

class geosubNew(SageObject):
    
    def __init__(self, sigma, k, presuf, dual):
        
        self._sigma = WordMorphism(sigma)
        M = self._sigma.incidence_matrix()
        b1 = max(M.eigenvalues())
        f = b1.minpoly()
        K = NumberField(f, 'b')
        b, = K.gens()
        #K.<b> = NumberField(M.characteristic_polynomial())
        #self._matrix() = M
        
        self._d = len(sigma.keys())
        self._k = k
        self._dual = dual
        self._field = K    
        
        S = sigma.keys() 
        alph = [x for x in tuples(S,k) if all(x[i] < x[i+1] for i in range(k-1))]
        
        X = {}
        E = geosub(sigma,k,presuf,dual)
        
        for a in alph:
            P = kPatch([kFace(zero_vector(self._d),a)])
            X[a] = list(newthing(newthing(E(P,1),0),0))    #  double newthing if necessary
    
        self._base_iter = X
              
    def __call__(self, patch, iterations=1):
    
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
        return (kFace(x_new + (-1)^(self._dual)*y.vector(), y.type(), y.mult(), D) for y in self._base_iter[t])    
    
    def __repr__(self):
        
        if self._dual==0: return "modified E_%s(%s)" % (self._k,str(self._sigma))
        else: return "modified E*_%s(%s)" % (self._k,str(self._sigma))
def Fcoord(face):
    base = face.vector()
    sh = face.type()
    return [base,base+abel([sh[0]]),base + abel([sh[1]]),base + abel([sh[0],sh[1]])]

def findexagons(L):
    Lfin = []
    for i in range(len(L)):
        for j in range(i+1,len(L)):
            for k in range(j+1,len(L)):
                A1 = set([tuple(x) for x in Fcoord(L[i])])
                A2 = set([tuple(x) for x in Fcoord(L[j])])
                A3 = set([tuple(x) for x in Fcoord(L[k])])
                if len(A1.intersection(A2)) == 2 and len(A1.intersection(A3)) == 2 and len(A2.intersection(A3)) == 2:
                    Lfin.append([L[i],L[j],L[k]])
    return Lfin  

def flip(patch):
    F1 = patch[0] 
    F2 = patch[1] 
    F3 = patch[2]
    a1 = set([tuple(x) for x in Fcoord(F1)])
    a2 = set([tuple(x) for x in Fcoord(F2)])
    a3 = set([tuple(x) for x in Fcoord(F3)])
    Pcom = a1.intersection(a2).intersection(a3)
    el1 = a2.intersection(a3) - a1.intersection(a2).intersection(a3)
    el2 = a1.intersection(a3) - a1.intersection(a2).intersection(a3)
    el3 = a1.intersection(a2) - a1.intersection(a2).intersection(a3)
    t1 = vector(list(el1)[0])-vector(list(Pcom)[0])
    t2 = vector(list(el2)[0])-vector(list(Pcom)[0])
    t3 = vector(list(el3)[0])-vector(list(Pcom)[0])
    return kPatch([kFace(F1.vector() + t1,F1.type(),F1.mult()),kFace(F2.vector() + t2,F2.type(),F2.mult()),kFace(F3.vector() + t3,F3.type(),F3.mult())])

def newthing(patch,force):
    
    Ex = findexagons(list(patch))
    P = patch
    for y in Ex:
        if len(set([y[0].type(),y[1].type(),y[2].type()])) == 3:
            Pnew = P + kPatch([kFace(face.vector(),face.type(),-face.mult(),face.dual()) for face in y]) + flip(y)
            if force == 0:
                if len(Pnew) < len(P):
                    P = Pnew
            else:
                P = Pnew        
    return P
