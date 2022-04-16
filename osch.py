import numpy as np
import pandas as pd
import op
import calendar
import ml

path='D:\\ml\\'


def reverse(x=np.array([])):
    n=int(x.shape[0]/2)
    i=0
    k=-1
    while i<n:
        a=x[i]
        b=x[k]
        x[i]=b
        x[k]=a
        i+=1
        k-=1
    return x

def hev(x):
    if x==0: return x
    if x>0: return 1
    else: return -1

class queues:
    def __init__(self,queue=np.array([])):
        self.i=1
        self.queue=queue
    def get_values(self,i=0):
        try:
            x=self.queue[:,i]
            return x
        except IndexError:
            return None

class debit_function:
    def __init__(self,q0=1,q1=1,tau=0,t=0):
        self.q0=q0
        self.q1=q1
        self.tau=tau
        self.t=t
        self.teta=self.q1*(self.t-self.tau)
        self.dq=self.q1-self.q0
        self.b=self.teta-self.dq*self.t
        self.sign=hev(self.dq)
        self.ro=1
        self.supp=np.array((0,self.t),dtype=np.float32)
        #self.cs -compact support
        self.cs=False
        self.index=0
        self.current_index = 0
        self.order=0
        self.gamma=0.01
        self.epsilon=tau
        self.service=np.array([],dtype=np.int32)
        self.equipment = np.array([], dtype=np.int32)
        #self.busy=False
        self.t1=np.NINF
        self.t2=np.NINF
        self.x1=np.NINF #временные координаты входа/выхода
        self.x2=np.NINF
        self.span=(np.NINF,np.inf)
        self.key=None
        self.used=False
        self.prohibits=np.array([])
        if self.dq<0:
            self.ro=0
        self.drift=self.b*self.ro+self.teta*(1-self.ro)

    def isbusy(self,x):
        if (x>=self.t1)&(x<=self.t2):
            return True
        else:
            return False


    def value(self,x):
        return self.teta-self.dq*x

    def tail(self,x):
        b=self.t-self.tau
        a=b-self.epsilon
        w=self.scaled_v1(a)
        if x>a:
            y=(a-x)/(b-x)
            return w+w*((1./np.exp(-self.gamma*y**2))-1)
        elif x>b:
            return np.inf
        else:
            return self.scaled_v1(x)


    def scaled(self,x):
        return (self.value(x)-self.drift)*self.sign

    def scaled_v1(self,x):
        return self.dq*(self.t-self.tau-x)

    def scaled_v2(self,x):

        scaled=self.scaled_v1(self.t-self.tau-self.epsilon)
        return scaled+self.tail(x)

    def update(self):
        self.teta = self.q1 * (self.t - self.tau)
        self.dq = self.q1 - self.q0
        self.b = self.teta - self.dq * self.t
        self.sign = hev(self.dq)
        self.drift = self.b * self.ro + self.teta * (1 - self.ro)
        if self.dq<0:
            self.ro=0

class values:
    def __init__(self):
        self.value=np.NINF
        self.index=0
        self.inf=False
        self.t1=np.NINF
        self.t2=np.NINF

class wells_schedule:
    def __init__(self):
        self.trajectories=[]
        #self.wells=np.array([])#номера скважин
        self.free=np.array([]) #номера скважин
        self.unical = np.array([]) #уникальные номера скважин
        self.cd=0 #текущее значение дебита (current_debit)
        self.groups=np.array([])#начальное распределение бригад по скважинам
        #self.group_index=np.arange(self.group.shape[0]) # индексация бригад
        self.numbers=np.array([])# индексация бригад
        self.ct=np.array([]) #время освобождения бригад (current time)
        self.dt = np.array([])  # дополнительное время
        self.st=np.array([]) #время начала работы на скважине (start time)
        self.ts=np.array([]) #транспортная матрица
        self.service=np.array([])#номера сервиса
        self.equipment=np.array([])  # номера сервиса
        self.wells_service=np.array([])#номера сервиса для скважин
        self.wells_equipment=np.array([])  # номера оборудования для скважин
        self.tr=np.array([])#время продолжительности ремонта
        self.Q0=np.array([])#дебит до ремонта
        self.Q1=np.array([])#дебит после ремонта
        self.weights = np.array([])  # матрица рейтингов скважин
        self.fun=self.f1
        self.values=np.array([])
        self.logistic_values = None
        self.queue=dict({}) #словарь с значениями key - номер скважины, value -
        self.vector = np.array([],dtype=bool) #вектор состояния скважин для бригады. True - значение конечное
        self.tracing=False
        self.t=100 #время на проведение мероприятий
        self.debit_functions=dict({})
        self.epsilon=np.inf
        self.minct=0.
        self.maxct=0.
        self.delta=1.
        self.ftmatrix = np.array([],dtype=bool)
        self.group_support = None
        self.transpose=False
        self.function=op.get_optim_trajectory
        self.routes=None
        self.start=None
        self.end=None

    def fit(self,ts,tr,Q0,Q1,groups,support=None,used=None,stop=None,queue=dict({}),epsilon=np.inf,service=np.array([]),equipment=np.array([]),wells_service=None,wells_equipment=None,prohibits=None,delta=3.,group_support=None,tracing=None):
        def update_df():
            if self.routes is not None:
                for i in np.arange(self.routes.shape[1]):
                    for j in np.arange(self.routes.shape[0]):
                        well=self.routes[j,i]
                        if well<0:
                            break
                        try:
                            (span)=self.set_span(j,i)
                            self.debit_functions[well].t1=self.start[j,i]
                            self.debit_functions[well].t2=self.end[j,i]
                            self.debit_functions[well].span=span
                        except KeyError:
                            continue

        #used - булев массив размерности groups. True - индекс бригады, размещенной на скважине для ремонта. False - бригада на скважине, но на работу не назначена.
        if (tracing is not None) and isinstance(tracing,bool):
            self.tracing=tracing
        self.Q1=Q1
        self.Q0=Q0
        self.dQ=self.Q1-self.Q0
        self.dq=np.min(self.dQ)
        self.groups=groups
        if used is None:
            used=np.zeros(self.groups.shape[0],dtype=bool)
        self.ts=ts
        self.tr=tr
        self.delta=delta
        self.free=np.arange(Q0.shape[0])
        self.mask=np.ones(Q0.shape[0],dtype=bool)
        self.cd=0
        self.queue=queue
        self.numbers=np.arange(groups.shape[0])
        self.ct=np.zeros(self.groups.shape[0],dtype=np.float)
        self.dt = np.zeros(self.groups.shape[0],dtype=np.float)
        self.current_index = np.zeros(self.groups.shape[0],dtype=np.int32)
        self.epsilon=epsilon
        self.service=service
        self.equipment=equipment
        self.wells_service=wells_service
        self.wells_equipment=wells_equipment
        self.ftmatrix=np.ones(shape=(self.Q0.shape[0],self.groups.shape[0]),dtype=bool)
        self.prohibits=prohibits

        if group_support is not None:
            self.group_support=group_support

        if used.shape[0]>0:
            mask=~np.isin(self.free,self.groups[used])
            self.free = self.free[mask]
            ext=support[self.groups[used],0]
            submask=np.isnan(ext)
            ext[submask]=0
            tv = self.tr[self.groups[used]]+ext

            mask=tv<0
            tv[mask]=0
            self.ct[used]=tv

        self.minct=self.ct.min()
        self.maxct=self.ct.max()
        self.st=np.zeros(self.groups.shape[0],dtype=np.float)
        self.values=np.empty(self.Q0.shape[0])
        self.values.fill(np.nan)
        if stop is not None:
            self.stop=stop
        else:
            self.stop=self.Q0.shape[0]
        self.maxcs=np.NINF
        self.mincs=np.inf
        self.supported=np.array([])
        self.support = support
        if support is None:
            self.support=np.empty(shape=(self.Q0.shape[0],2))
            self.support.fill(np.nan)

        maxsc=np.NINF
        minsc=np.inf
        supported=[]

        for i in self.free:
            supp=self.support[i]
            wf=debit_function(self.Q0[i],self.Q1[i],self.tr[i],self.t)
            wf.index=i
            if self.prohibits is not None:
                proht=self.prohibits[i]
                wf.prohibits=np.where(proht==False)[0]

            if self.wells_equipment is not None:
                wf.equipment=self.wells_equipment[i]

            if self.wells_service is not None:
                wf.service=self.wells_service[i]

            mask=np.isnan(supp)
            #mark=False
            for j,m in enumerate(mask):
                if ~m:
                    wf.supp[j]=supp[j]
                    wf.cs=True
            if wf.cs:
                supported.append(i)
                mina=wf.scaled_v1(wf.supp[0])
                minb=wf.scaled_v1(wf.supp[1])
                mins=min(mina,minb)
                if minsc>mins:
                    minsc=mins
            self.debit_functions.update({i:wf})
            for n in self.numbers:
                self.ftmatrix[i,n]=self.isvalid(n,i)

            if self.tracing:
                val=wf.supp[0]
            else:
                val=wf.scaled_v1(0)
            self.values[i]=val

            if maxsc<self.values[i]:
                maxsc=self.values[i]

        self.supported=np.array(supported)
        mask=~np.isnan(self.values)
        self.values=self.values[mask]
        self.sindices=np.argsort(self.values)
        if not self.tracing:
            self.sindices=reverse(self.sindices)
        self.free=self.free[self.sindices]
        self.infmask=np.zeros(shape=self.stop,dtype=bool)
        self.infindex = np.zeros(shape=self.stop, dtype=np.int32)
        self.nempty = np.arange(self.groups.shape[0])
        self.maxcs=maxsc
        self.mincs=minsc
        self.prohib_dict = self.get_prohib_dict()
        update_df()

    def get_group_support(self,x=0.,i=0):
        # устанавливает значение x в соответствии с ограничениями на время работы группы
        if self.group_support is None:
            return x
        hour_,day=np.modf(x)
        #hour_=hour*24
        try:
            support=self.group_support[i]
            if (hour_>=support[0])&(hour_<=support[1]):
                return x
            elif hour_<support[0]:
                return day+support[0]
            else:
                a=1+support[0]
                return day+a

        except IndexError:
            return x


    def update_debit_functions(self,debitfun_dict=dict({})):
        for i in debitfun_dict.items():
            self.debit_functions.update({i})

    def get_prohib_dict(self):
        if self.prohibits is None:
            return dict()
        index = np.arange(self.prohibits.shape[0])
        frame = dict()
        shape=max(self.prohibits.shape)
        mask = np.ones(shape=shape, dtype=bool)
        k = 0
        i = 0
        while k < index.shape[0]:
            if mask[k]:
                indices = np.where(self.prohibits[k] == False)[0]
                mask[k] = False
                if indices.shape[0] > 0:
                    indices=np.append(k, indices)
                    frame.update({i: indices })
                    mask[indices] = False

                    for s in indices:
                        try:
                            fun=self.debit_functions[s]
                            fun.key = i
                        except KeyError:
                            continue


                    i += 1

            k += 1
        return frame

    def isprohibited(self,x=0,well=0):
        try:

            fun=self.debit_functions[well]

            for i in fun.prohibits:
                try:
                    if  self.debit_functions[i].isbusy(x)|self.debit_functions[i].isbusy(x+fun.tau):
                        return False
                except KeyError:
                    continue
            return True

        except KeyError:
            return False


    def get_weights(self, indices=np.array([0])):

        if self.stop > self.free.shape[0]:
            self.stop = self.free.shape[0]
            self.infmask = np.zeros(shape=self.stop, dtype=bool)
            self.infindex = np.zeros(shape=self.stop, dtype=np.int32)


        if self.free.shape[0] < self.groups.shape[0]:
            self.weights=np.array([])
            self.infmask = np.zeros(shape=self.groups.shape, dtype=bool)
            self.infindex = np.zeros(shape=self.groups.shape, dtype=np.int32)
            k=0
            for i in np.arange(self.free.shape[0]):
                v = self.get_vector_t(i, array=True)
                if k > 0:
                    self.weights = np.vstack((self.weights, v))
                else:
                    self.weights = v.copy()
                k+=1

            self.nempty = self.infindex[self.infmask]
            sargs = np.argsort(self.nempty)
            self.nempty=self.nempty[sargs]
            if self.nempty.shape[0] > 0:
                self.transpose=True

                return self.weights[self.nempty]
            else:
                return None
        #self.weights = np.empty(shape=(self.groups.shape[0], self.free.shape[0]))
        self.weights = np.empty(shape=(self.groups.shape[0], self.stop))
        tolerance=self.epsilon
        missed=[]
        forw=True
        while forw:
            for i in indices:
                ct = self.ct[i]
                if ct - self.minct < tolerance:
                    v = self.get_vector(i, array=True)
                    self.weights[i] = v
                else:
                    missed.append(i)

            self.nempty = self.infindex[self.infmask]
            sargs = np.argsort(self.nempty)
            self.nempty=self.nempty[sargs]

            if self.nempty.shape[0] > 0:
                return np.array(self.weights)[self.nempty]

            else:
                if tolerance <= self.maxct - self.minct:
                    tolerance+=self.delta
                else:
                    forw=False
                if len(missed)>0:
                    indices=np.array(missed)
                    missed=[]
                else:
                    return None

        return None


    def get_span(self, i=0):

        if self.routes is None:
            return np.inf, np.inf
        if i >= self.routes.shape[1]:
            return np.NINF, np.inf

        ci = self.current_index[i]

        if ci >= self.routes.shape[0]:
            return np.NINF, np.inf

        target = self.routes[ci, i]

        if target < 0:
            return np.NINF, np.inf

        t1=self.debit_functions[target].span[0]
        t2 =self.debit_functions[target].span[1]


        return t1, t2

    def get_span_v1(self,i=0):

        if self.routes is None:
            return np.inf,np.inf
        if i >= self.routes.shape[1]:
            return np.NINF, np.inf

        ci = self.current_index[i]

        if ci >= self.routes.shape[0]:
            return np.NINF, np.inf

        target = self.routes[ci, i]

        if target<0:
            return np.NINF, np.inf

        t1 = self.start[ci, i]
        t2 = self.end[ci, i]
        if ci < self.routes.shape[0] - 1:
            next_target = self.routes[ci + 1, i]
            if next_target>=0:
                t3 = self.start[ci + 1, i]
                ts = self.ts[target, next_target]
            else:
                t3 = np.inf
                ts = np.inf
        else:
            t3 = np.inf
            ts = np.inf
        fun=self.debit_functions[target]
        bound = fun.supp[1]
        delta = get_delta(t1, t2, t3, ts=ts, bound=bound)
        tau=np.inf

        for i in fun.prohibits:
            fun_=self.debit_functions[i]
            if (not fun_.used) and (fun_.cs) and (fun_.supp[0]>=t2):
                tau_=fun_.supp[0]-t2
                if tau_<tau:
                    tau=tau_
        delta=min(delta,tau)
        return t1,t1+delta

    def set_span(self,ci=0,i=0):

        if self.routes is None:
            return np.inf,np.inf
        if i >= self.routes.shape[1]:
            return np.NINF, np.inf

        target = self.routes[ci, i]

        if target<0:
            return np.NINF, np.inf

        t1 = self.start[ci, i]
        t2 = self.end[ci, i]
        if ci < self.routes.shape[0] - 1:
            next_target = self.routes[ci + 1, i]
            if next_target>=0:
                t3 = self.start[ci + 1, i]
                ts = self.ts[target, next_target]
            else:
                t3 = np.inf
                ts = np.inf
        else:
            t3 = np.inf
            ts = np.inf
        fun=self.debit_functions[target]
        bound = fun.supp[1]
        delta = get_delta(t1, t2, t3, ts=ts, bound=bound)
        tau=np.inf

        for i in fun.prohibits:
            fun_=self.debit_functions[i]
            if (not fun_.used) and (fun_.cs) and (fun_.supp[0]>=t2):
                tau_=fun_.supp[0]-t2
                if tau_<tau:
                    tau=tau_
        delta=min(delta,tau)
        return t1,t1+delta

    def finit(self,a=np.NINF,b=np.inf):
        def function(fun,x=0.):
            if (x>=a)&(x<=b):
                return self.fun(x=x,fun=fun)
            else:
                return np.NINF
        return function


    def check_infty(self,v=np.array([]),gi=0):
        indices=np.arange(self.vector.shape[0])

        index=indices[self.vector]
        infty=indices[~self.infmask]
        self.nempty = self.infindex[self.infmask]
        stop=False
        for i in index:
            x=v[i]
            ihat=self.infindex[i]
            for j in infty:
                y=self.weights[ihat,j]
                if ~np.isinf(y):
                    self.infmask[j]=True
                    self.infindex[j]=ihat
                    self.infindex[i]=gi
                    stop=True
                    break
            if stop:
                break
        if not stop:
            for i in index:
                x=v[i]
                ihat = self.infindex[i]
                y = self.weights[ihat, i]
                if x>y:
                    self.infindex[i]=gi
                    break



        #return x


    def get_vector(self, i=0, array=True):
        values = []
        cw = self.groups[i]
        k = 0
        inf = True
        # i- индекс бригады
        a_, b_ = self.get_span(i)
        fun = self.finit(a_, b_)
        next_well = None
        cv = np.inf
        dt = 0.
        ck = None
        self.vector=np.zeros(shape=self.stop,dtype=bool)
        # fun()
        if ~np.isinf(a_):
            # wi- индекс следующей по расписанию скважины
            # next_well- номер следующей по расписанию скважины
            wi = self.current_index[i]
            next_well = self.routes[wi, i]

        while k < self.stop:
            j = self.free[k]

            a = self.ts[cw, j]
            x = a + self.ct[i]


            x=self.get_group_support(x,i)

            free_=self.isprohibited(x,j)


            if (self.ftmatrix[j,i])& free_:
                func=self.debit_functions[j]
                func.current_index=k

                if next_well is not None:
                    if j == next_well:
                        ck = k
                        dt = a_ - x

                        value = fun(fun=func, x=x)


                    else:
                        hat = x + self.tr[j] + self.ts[j, next_well]
                        hat = self.get_group_support(hat, i)
                        if hat <= b_:
                            if not func.cs:
                                value = self.fun(x=x, fun=func)

                            else:
                                value = np.NINF
                        else:
                            value = np.NINF

                else:
                    if self.tracing:
                        value = self.fun(x=x, fun=func)
                    else:
                        if not func.cs:
                            value = self.fun(x=x, fun=func)

                        else:
                            value = np.NINF

            else:
                value = np.NINF


            if ~np.isinf(value):
                self.vector[k]=True


            if inf & ~np.isinf(value):

                if ~self.infmask[k]:
                    self.infmask[k] = True
                    self.infindex[k] = i
                    inf = False

            values.append(value)
            k += 1

        if inf:
            if (next_well is not None) and (ck is not None):
                k = ck

                self.infmask[k] = True
                self.infindex[k] = i
                inf = False
                values[k] = cv
                self.dt[i] = dt

            else:
                self.check_infty(np.array(values),i)

        if array:
            return np.array(values).reshape(1,-1)
        else:
            return values

 # вычисление вектора состояний для случая, когда количество бригад больше, чем количество скважин
    def get_vector_t(self,i=0,array=True):
        values=[]
        cw=self.free[i]
        func=self.debit_functions[cw]
        func.current_index=i
        k=0
        inf=True
        cv = np.inf
        ck = None
        dt=0.
        nw = None
        self.vector = np.zeros(shape=self.groups.shape[0], dtype=bool)


        while k<self.groups.shape[0]:
            j=self.groups[k]
            a = self.ts[j, cw]
            x = a + self.ct[k]
            x=self.get_group_support(x,k)
            free_ = self.isprohibited(x, j)

            if self.ftmatrix[cw,k]&free_:

                # k- индекс бригады
                a_, b_ = self.get_span(k)
                fun = self.finit(a_, b_)
                next_well = None

                if ~np.isinf(a_):
                    # wi- индекс следующей по расписанию скважины
                    # next_well- номер следующей по расписанию скважины
                    wi = self.current_index[k]
                    next_well = self.routes[wi, k]


                if next_well is not None:
                    if cw == next_well:
                        ck = k
                        dt = a_ - x
                        nw = next_well
                        value = fun(fun=func, x=x)
                    else:
                        hat = x + self.tr[cw] + self.ts[cw, next_well]
                        hat = self.get_group_support(hat, k)
                        if hat <= b_:
                            if not func.cs:
                                value = self.fun(x=x, fun=func)
                            else:
                                value = np.NINF
                        else:
                            value = np.NINF

                else:
                    if self.tracing:
                        value = self.fun(x=x, fun=func)
                    else:
                        if not func.cs:
                            value = self.fun(x=x, fun=func)
                        else:
                            value = np.NINF
            else:
                value = np.NINF



            if ~np.isinf(value):
                self.vector[k]=True

            if inf&~np.isinf(value):

                if ~self.infmask[k]:
                    self.infmask[k]=True
                    self.infindex[k]=i
                    inf=False

            values.append(value)
            k+=1

        if inf:
            if nw is not None:
                k = ck
                self.infmask[k] = True
                self.infindex[k] = i
                inf = False
                values[k]=cv

                self.dt[k]=dt
            else:
                self.check_infty(np.array(values), i)

        if array:
            return np.array(values).reshape(1,-1)
        else:
            return values


    def f0(self,x=0.,fun=debit_function()):
        return fun.value(x)

    def f1(self,x=0.,fun=debit_function()):
        return fun.value(x)*fun.dq

    def f2(self,x=0.,fun=debit_function()):
        # ранжирование по дебиту. Формула Елина Н.Н.
        return (fun.q0*x+fun.q1*(self.t-fun.tau-x)-(fun.q0+self.dq)*self.t)/self.t

    def f3(self,x=0.,fun=debit_function()):
        return (fun.value(x)-fun.drift)*fun.dq

    def f4(self,x=0.,fun=debit_function()):
        return (fun.value(x)-fun.drift)*fun.sign

    def f5(self,x=0.,fun=debit_function()):
        return -fun.dq*x+fun.dq*self.t-fun.q1*fun.tau-self.dq*self.t

    def f6(self,x=0.,fun=debit_function()):
        # ранжирование по дебиту
        if (x>=fun.supp[0])&(x<=fun.supp[1]):
            return fun.dq*(self.t-fun.tau-x)
        else:
            return np.NINF

    def f7(self,x=0.,fun=debit_function()):
        teta=fun.supp[0]-x
        if teta>=0:
            return -fun.supp[0]
        else:
            teta = x
            if teta<=fun.supp[1]:
                return -teta
            else:
                return np.NINF

    def f8(self,x=0.,fun=debit_function()):
        teta=fun.supp[1]-x
        if teta>=0:
            return -teta
        else:
            return np.NINF

    def f9(self,x=0.,fun=debit_function()):
        teta=-self.f7(x,fun=fun)
        d=-(teta+fun.tau)*(fun.supp[1]-teta)/(fun.supp[1]-fun.supp[0])
        return d

    def f10(self,x=0.,fun=debit_function()):
        teta=-self.f7(x,fun=fun)
        d=-(teta+fun.tau)/(fun.supp[1]-fun.supp[0])
        return d

    def f11(self,x=0.,fun=debit_function()):
        teta=-self.f7(x,fun=fun)
        d=teta+(fun.tau/(fun.supp[1]-fun.supp[0]))
        return -d


    def f12(self,x=0.,fun=debit_function()):
        teta=-self.f7(x,fun=fun)
        if teta>fun.supp[1]:
            return np.NINF
        d=teta+(fun.supp[1]-teta)/fun.tau
        return -d

    def f13(self,x=0.,fun=debit_function()):
        #ранжирование по логистике
        # i - номер группы
        teta=-self.f7(x,fun=fun)
        if np.isinf(teta):
            return np.NINF
        #teta=self.get_group_support(teta,i)
        cw=fun.index
        t=teta+fun.tau
        #t = self.get_group_support(t, i)
        d=0
        k=0
        n=0
        sum=0
        while k < self.stop:
            j = self.free[k]
            if j!=cw:
                a = self.ts[cw, j]
                teta_=t+a
                #teta_ = self.get_group_support(teta_, i)
                value=teta_-self.debit_functions[j].supp[1]
                sum+=value
                n+=1

            k += 1

        if n>0:
            return -sum/n
        else:
            return 0

    def f14(self,x=0.,fun=debit_function()):
        #ранжирование по логистике
        # i - номер группы
        def set_log_values():
            stop=self.free.shape[0]
            self.logistic_values=np.zeros(shape=stop)
            i=0
            while i<self.logistic_values.shape[0]:
                cw=self.free[i]
                k = 0
                sum=0.
                n=0
                while k < stop:
                    j = self.free[k]
                    if j != cw:
                        value =self.ts[cw, j] - self.debit_functions[j].supp[1]
                        sum += value
                        n += 1
                    k += 1
                #if n > 0:
                self.logistic_values[i]=sum

                i+=1

        if self.logistic_values is None:
            set_log_values()
            value=self.f14(x,fun)
            return value
        else:
            teta = -self.f7(x, fun=fun)
            if np.isinf(teta):
                return np.NINF
            cw = fun.current_index
            t = teta + fun.tau
            n=self.free.shape[0]-1
            if n>0:
                return -(t+self.logistic_values[cw]/n)
            else: return -t

    def update_logistic_values(self, indices=np.array([])):
        def get_values(i=0,indices=np.array([])):
            sum=0
            for j in indices:
                if i==j:
                    return None
                value = self.ts[i, j] - self.debit_functions[j].supp[1]
                sum += value
            return sum

        if (self.logistic_values is None)|(indices.shape[0]==0):
            return

        k=0
        mask=np.ones(shape=self.logistic_values.shape[0],dtype=bool)
        while k<self.logistic_values.shape[0]:
            j=self.free[k]
            value=get_values(i=j,indices=indices)
            if value is None:
                mask[k]=False
            else:
                self.logistic_values[k]=self.logistic_values[k]-value
            k+=1
        self.logistic_values=self.logistic_values[mask]



    def optim(self,t):
        s=0
        for k in self.debit_functions.keys():
            fun=self.debit_functions[k]
            if fun.dq>=0:
                s+=fun.value(0)
            else:
                s+=fun.value(t)
        return s

    def get_queue(self,i,j=None):
        current=0
        try:
            well=self.queue[i]
            if j is not None:
                current=j
            else:
                current=well.i
            data=well.get_values(current)
            well.i+=1
            return data,current
        except KeyError:
            return None,current


    def get_optimized_trajectories(self,indices=np.array([0])):
        vectors=self.get_weights(indices=indices)
        if vectors is  None:
            return None

        try:

            taken,s=self.function(vectors,criterion='max')

        except (IndexError):
            return None



        if self.transpose:
            index=self.nempty.copy()
            self.nempty=taken[1].copy()
            taken[1]=index

        mask=self.block(taken)
        taken=taken[:,mask]
        self.nempty=self.nempty[mask]

        if self.transpose:
            self.transpose = False

        return taken,s

    def block(self,indices=np.array([])):

        def get_keys(indices=np.array([])):
            code = np.empty(shape=indices.shape[1], dtype=np.int32)
            code.fill(-1)
            #code=[]
            k = 0
            while k < indices.shape[1]:
                i = indices[1][k]
                w=self.free[i]
                fun = self.debit_functions[self.free[i]]
                if fun.key is not None:
                    code[k]=fun.key
                k += 1
            return code

        def get_separated(y=np.array([])):
            mask = np.ones(shape=y.shape[1], dtype=bool)
            k = 0
            index = [1, 2]
            values = y[0]
            sindex = reverse(np.argsort(values))
            x = y[:, sindex]
            i = np.arange(mask.shape[0])[sindex]
            while k < mask.shape[0]:
                if mask[k]:
                    t = x[index, k]
                    j = k + 1
                    while j < mask.shape[0]:
                        if mask[j]:
                            t_ = x[index, j]
                            interseption = op.interseption(t, t_,shape=2)
                            if interseption.shape[0] > 0:
                                mask[j] = False

                        j += 1
                k += 1
            return i[~mask], i[mask]

        mask = np.ones(shape=indices.shape[1], dtype=bool)
        keys=get_keys(indices)

        uniques,counts=np.unique(keys,return_counts=True)
        x_index = self.nempty[indices[0]]  # -номера строк матрицы self.weight
        y_index = self.groups[x_index]  # -номера скважин, на которых находятся бригады x_index

        if self.transpose:
            values = self.weights.T[x_index, indices[1]]
        else:
            values = self.weights[x_index, indices[1]]

        m_index=self.free[indices[1]]
        #m_index=self.unical[indices[1]]

        k=0
        while k<counts.shape[0]:
            c=counts[k]
            if c>1:
                key=uniques[k]
                if key>=0:
                    index=np.where(keys==key)[0]
                    vector=np.empty(shape=(3,index.shape[0]))
                    for i, j in enumerate(index):
                        val = values[j]
                        w = m_index[j]
                        cw = y_index[j]
                        ci = x_index[j]
                        t1_ = self.ct[ci] + self.ts[cw, w] + self.dt[ci]
                        t2_ = t1_ + self.tr[w]
                        vector[0, i] = val
                        vector[1, i] = t1_
                        vector[2, i] = t2_

                    get_out, leave = get_separated(vector)
                    mask[index[get_out]] = False


            k+=1
        return mask


    def update(self,indices=np.array([]),s=0):
        index=self.free[indices[1]]
        self.update_schedule(index=index)
        self.cd=self.cd+s
        if index.shape[0]<self.groups.shape[0]:
            index_x=np.empty(self.groups.shape[0], dtype=int)
            index_y=np.empty(self.groups.shape[0], dtype=np.float32)
            index_z=np.empty(self.groups.shape[0], dtype=np.float32)
            index_w = np.zeros(self.groups.shape[0], dtype=int)
            index_x.fill(-1)
            index_y.fill(-1)
            index_z.fill(-1)
            sa=np.argsort(self.nempty)
            #si=index[sa]
            #sne=self.nempty[sa]
            index_x[self.nempty[sa]] = index[sa]

            for i,w in enumerate(index):
                j=self.nempty[i]
                st=self.ct[j] + self.ts[self.groups[j], w]+self.dt[j]
                self.st[j] =self.get_group_support(st,j)
                self.dt[j]=0
                self.ct[j] = self.st[j] + self.tr[w]
                fun=self.debit_functions[w]
                fun.used=True
                fun.t1=self.st[j]
                fun.t2=self.ct[j]

            index_y[:]=self.st
            index_z[:]=self.ct
            #index=index_x
        else:
            index_w = np.zeros(index.shape[0], dtype=int)
            for i,w in enumerate(index):
                j=self.nempty[i]
                st=self.ct[j] + self.ts[self.groups[j], w]+self.dt[j]
                self.st[j] =self.get_group_support(st,j)
                #self.st[j] = self.ct[j] + self.ts[self.groups[j], w]+self.dt[j]
                self.ct[j] = self.st[j] + self.tr[w]
                self.dt[j] = 0
                fun=self.debit_functions[w]
                fun.used=True
                fun.t1=self.st[j]
                fun.t2=self.ct[j]

            index_y=self.st.copy()
            index_z=self.ct.copy()
            index_x=index[self.nempty]

        self.groups[self.nempty]=index
        self.mask=np.ones(self.free.shape[0],dtype=bool)
        self.mask[indices[1]]=0
        #self.mask=self.wells[1]==1
        self.free=self.free[self.mask]
        self.infmask = np.zeros(shape=self.stop, dtype=bool)
        self.infindex = np.zeros(shape=self.stop, dtype=np.int32)
        self.minct = self.ct.min()
        self.maxct=self.ct.max()
        #self.free=self.sindices[self.mask]
        #self.trajectories.append([index,(self.st,self.ct)])
        self.trajectories.append([index_x,(index_y,index_z,index_w)])

    def update_schedule(self,index=np.array([])):
        if self.routes is not None:
            k=0
            for i in self.nempty:
                ci=self.current_index[i]
                well=self.routes[ci,i]
                taken=index[k]
                if taken==well:
                    self.current_index[i]=ci+1
                k+=1

    def update_queue(self,w=0,i=None):
        queue,current = self.get_queue(w,i)
        if queue is not None:
            fun = self.debit_functions[w]
            fun.supp = queue[[0, 1]]
            fun.tau = queue[2]
            fun.q0 = queue[3]
            fun.q1 = queue[4]
            fun.update()
            self.tr[w] = queue[2]
            self.Q0[w] = queue[3]
            self.Q1[w] = queue[4]
            self.dQ[w] = queue[4] - queue[3]
            return True, current
        else:
            return False,current




    def update_tracing(self,indices=np.array([]),s=0):
        index=self.free[indices[1]]
        self.update_schedule(index=index)
        self.mask=np.ones(self.free.shape[0],dtype=bool)
        self.mask[indices[1]]=0
        #print(index)
        self.cd=self.cd+s
        if index.shape[0]<self.groups.shape[0]:
            index_x=np.empty(self.groups.shape[0], dtype=int)
            index_y=np.empty(self.groups.shape[0], dtype=np.float32)
            index_z=np.empty(self.groups.shape[0], dtype=np.float32)
            index_w = np.zeros(self.groups.shape[0], dtype=int)
            index_x.fill(-1)
            index_y.fill(-1)
            index_z.fill(-1)
            #index_w.fill(0)
            index_x[self.nempty] = index

            for i,w in enumerate(index):
                j=self.nempty[i]
                point=self.ct[j]+self.ts[self.groups[j],w]
                #point=self.get_group_support(point)
                fun=self.debit_functions[w]
                if point<=fun.supp[0]:
                    st=fun.supp[0]
                    st=self.get_group_support(st,j)
                    self.st[j]=st
                    self.ct[j] = self.st[j]+self.tr[w]
                    fun.used = True
                    fun.t1 = self.st[j]
                    fun.t2 = self.ct[j]
                else:
                    if point<=fun.supp[1]:
                        st=self.get_group_support(point,j)
                        self.st[j] = st
                        self.ct[j] = self.st[j] + self.tr[w]
                        fun.used = True
                        fun.t1 = self.st[j]
                        fun.t2 = self.ct[j]
                    else:
                        self.st[j] = np.nan
                        self.ct[j] = np.nan

                state,current=self.update_queue(w)
                if current>0:
                    index_w[j] = current-1

                if state:
                    s = indices[1][i]
                    self.mask[s]=1


            index_y[:]=self.st
            index_z[:]=self.ct
            #index=index_x
        else:

            index_w = np.zeros(index.shape[0], dtype=int)
            for i,w in enumerate(index):

                j=self.nempty[i]
                point=self.ct[j]+self.ts[self.groups[j],w]
                fun=self.debit_functions[w]
                if point<=fun.supp[0]:
                    st=fun.supp[0]
                    self.st[j]=self.get_group_support(st,j)
                    self.ct[j] = self.st[j]+self.tr[w]
                    fun.used = True
                    fun.t1 = self.st[j]
                    fun.t2 = self.ct[j]
                else:
                    if point<=fun.supp[1]:
                        st=self.get_group_support(point,j)
                        self.st[j] = st
                        self.ct[j] = self.st[j] + self.tr[w]
                        fun.used = True
                        fun.t1 = self.st[j]
                        fun.t2 = self.ct[j]
                    else:
                        self.st[j] = np.nan
                        self.ct[j] = np.nan

                state, current = self.update_queue(w)
                if current>0:
                    index_w[j] = current-1

                if state:
                    s=indices[1][i]
                    self.mask[s]=1


            index_y=self.st.copy()
            index_z=self.ct.copy()
            index_x=index[self.nempty]

        self.groups[self.nempty]=index
        self.update_logistic_values(index)
        self.free=self.free[self.mask]
        self.infmask = np.zeros(shape=self.stop, dtype=bool)
        self.infindex = np.zeros(shape=self.stop, dtype=np.int32)
        self.minct=self.ct.min()
        self.maxct = self.ct.max()
        self.trajectories.append([index_x,(index_y,index_z,index_w)])


    def isvalid(self,igroup,well):
        fun=self.debit_functions[well]
        service=self.service[igroup]
        equipment=self.equipment[igroup]
        serv=True
        equip=True
        for j in fun.service:
            val=service[j]
            if val==0:
                serv=False
                break
        for e in fun.equipment:
            val=equipment[e]
            if val==0:
                equip=False
                break

        return serv&equip



    def get_routes(self):

        while self.free.shape[0]>0:
            res=self.get_optimized_trajectories(self.numbers)
            if res is None:
                break
            indices=res[0]
            s=res[1]

            if self.tracing:
                self.update_tracing(indices,s)
            else:
                self.update(indices,s)
        return self.trajectories



def issubset (A=np.array([]),B=np.array([])):
    ax=A[0]
    ay=A[1]
    bx=B[0]
    by=B[1]
    mask=(bx>=ax)&(by<=ay)
    if mask:
        return True
    else:
        return False
def getsubsetmask(A=np.array([]),B=np.array([])):
    index=[]
    for i in np.arange(B.shape[1]):
        m=issubset(A[:,i],B[:,i])
        #index.append(m)
        if m:
            index.append(i)
    return np.array(index, dtype=int)

def get_sub_routes(w=0,t0=0,ts=np.array([]),ti=np.array([]), bounds=np.array([]),index=np.array([])):
    #print(index)
    left=ts[w,index[1]]+t0
    right=left+ti[index[1]]
    bnd=np.array([left,right])
    #print(bnd)
    #print(index[0])
    #print(bounds)
    indices=getsubsetmask(bounds[:,index[0]],bnd)
    return index[:,indices]

def getsub(w=np.array([]),t0=0,ts=np.array([]),ti=np.array([]), bounds=np.array([]),index=np.array([], dtype=int),mask=np.array([], dtype=bool),INDICES=[],trajectories=[], maxn=0, maxiter=np.inf):
    x0=t0
    indices=get_sub_routes(w=w,t0=x0,ts=ts,ti=ti, bounds=bounds,index=index[:,mask])
    if (indices.shape[1]>0)&(maxn<maxiter):
        maxn=maxn+1
    else:
        trajectories.append((INDICES,maxn,x0))
        return
    for i in np.arange(indices.shape[1]):
        t0=x0+ts[w,indices[:,i][1]]+ti[indices[:,i][1]]
        j=INDICES.copy()
        j.append(indices[:,i][0])
        mask[indices[:,i][0]]=False
        getsub(w=indices[:,i][1],t0=t0,ts=ts,ti=ti, bounds=bounds,index=index,INDICES=j,mask=mask, maxn=maxn,trajectories=trajectories, maxiter=maxiter)
        mask[indices[:,i][0]]=True
    return trajectories

def get_debit(wells=np.array([]),time_=np.array([]),wsch=wells_schedule(),mt=100):
    #mt -горизонт расчета
    s=np.empty(shape=wells.shape)
    s.fill(0.)
    for i in np.arange(wells.shape[1]):
        for j in np.arange(wells.shape[0]):
                w=wells[j,i]
                t=time_[j,i]
                if w>=0:
                    fun=wsch.debit_functions[w]
                    fun.t=mt
                    fun.teta = fun.q1 * (fun.t - fun.tau)
                    s_=fun.value(t)
                    s[j,i]=s_
    return s

def get_rout_time(routes):
    shape0=len(routes)
    shape1=routes[0][0].shape[0]
    route=np.ones(shape=(shape0,shape1),dtype=np.int32)*-1
    time = np.ones(shape=(shape0, shape1), dtype=np.float)*-1
    delta = np.ones(shape=(shape0, shape1), dtype=np.float)*-1
    order = np.ones(shape=(shape0, shape1), dtype=np.int32)*-1
    for i in np.arange(shape1):
        k=0
        for j in np.arange(shape0):
            ro = routes[j][0][i]
            if ro>=0:
                route[k,i]=ro
                time[k,i]=routes[j][1][1][i]
                delta[k,i]=routes[j][1][0][i]
                order[k, i] = routes[j][1][2][i]
                k+=1
    return route,time,delta,order


def debit(group=np.array([]),R=np.array([]),tr=np.array([]),ts=np.array([]),wsch=wells_schedule()):
    X=np.vstack((group,R))
    s=np.empty(shape=R.shape)
    for i in np.arange(X.shape[1]):
        x=X[:,i]
        t=0
        j=1
        while j<x.shape[0]:
            t=t+ts[x[j-1],x[j]]
            s_=wsch.debit_functions[x[j]].value(t)
            s[j-1,i]=s_
            t=t+tr[x[j]]
            j+=1

    return s

def get_month_range(month,year=2017):
    month=int(month)
    if month>9:
        month_=str(month)
    else:
        month_='0'+str(month)

    start=np.datetime64(str(year)+'-'+str(month_)+'-'+'01')
    last=calendar.monthrange(year,month)[1]
    end=np.datetime64(str(year)+'-'+str(month_)+'-'+str(last))
    return start,end

def get_quarter_range(quarter,year=2017):
    if quarter<1:
        return np.nan,np.nan
    quarter=int(quarter)
    a=int(quarter/4)
    b=np.fmod(quarter-1,4)
    c=np.array([1,2,3])
    #q=c[int(b)]-1
    year=year+a
    monthes=c+b*3
    start=get_month_range(monthes[0],year)[0]
    end=get_month_range(monthes[-1],year)[1]
    return start,end

def get_distances(data,fields,well='ID',field='Месторождение'):
    agg=data[[field,well]].groupby(well)
    first=agg.first()
    shape=first.shape[0]
    zeros=np.zeros(shape=(shape,shape))
    index=first.index

    for i in np.arange(first.shape[0]):
        f=first.iloc[i][0]
        #w=first.index[i]
        for j in np.arange(i+1,first.shape[0]):
            #w_=first.index[j]
            f_=first.iloc[j][0]
            fd=fields.loc[f,f_]
            zeros[i,j]=fd+6
            zeros[j,i]=fd+6
    return pd.DataFrame(data=zeros,index=index,columns=index)

def set_unical_well_index(wells=np.array([]),index=np.array([])):
    indices=np.zeros(wells.shape[0],dtype=np.int32)
    for i,j in enumerate(index):
        mask=np.where(wells==j)[0]
        indices[mask]=i
    return indices



def try_swap(a=np.array([]),b=np.array([]),funa=debit_function(),funb=debit_function(),l=0,bound=0,delta=0,eps=1e-3):
    suppa=funa.supp
    suppb=funb.supp
    interseption=op.interseption(suppa,suppb,shape=2)
    if interseption.shape[0]==0:
        return False
    s=funa.scaled_v1(a[0])+funb.scaled_v1(b[0])
    print(s)
    t1=interseption[0]
    t2=t1+b[1]-b[0]
    t3=t2+l
    t4=t3+a[1]-a[0]
    print(interseption)
    print(t1,t2,t3,t4)
    if abs(t4+delta-bound)<eps:
        s_ = funb.scaled_v1(t1) + funa.scaled_v1(t3)
        print(s_)
        if s_>s:
            return np.array([t1,t2]),np.array([t3,t4])
    return False

def get_delta(t1=0.,t2=0.,t3=0.,ts=0.,bound=0.):
    d1=t3-t2-ts
    d2=bound-t1
    delta=min(d2,d1)
    return delta


def get_queue(data,field='Скважина'):
    aggdata = data.groupby(field)
    first = aggdata.first()
    queue = dict({})
    for i, group in enumerate(aggdata):
        ID = group[0]
        p = group[1]
        # mask=p['Фиксация на месяц проведения']=='фиксация'
        mask = (~np.isnan(p['Start'])) & (p['Начатые операции'] != 'Начатое')
        q = p[mask]

        start = []
        end = []
        tau = []
        q0 = []
        q1 = []
        t = 0
        if q[mask].shape[0] > 1:
            for k in q.index:
                row = q.loc[k]
                #if t > 0:
                    # if row['Фиксация на месяц проведения']=='фиксация':
                start.append(row['Start'])
                end.append(row['End'])
                tau.append(row['Продолжительность ремонта'])
                q0.append(row['Stop oil rate'])
                q1.append(row['Start oil rate'])
                t += 1
        if t > 0:
            index = first.index.get_loc(ID)
            array = queues(np.array([start, end, tau, q0, q1]))
            queue.update({index: array})
    return first, queue


def get_otm_binary(otm_wells=np.array([]), otm_services=np.array([]), wells=pd.Index([]), services=np.array([]),
                   otm_prohibits=np.array([])):
    k = otm_wells.shape[0]
    n = k + wells.shape[0]
    binary = np.ones(shape=(otm_wells.shape[0], n), dtype=bool)

    i = 0
    while i < k:
        index = otm_wells[i]
        serv = otm_services[i]
        try:
            well = wells.get_loc(index)
            serv1 = services[well]
            for s in serv:
                for s_ in serv1:
                    if not otm_prohibits[s_, s]:
                        binary[i, k + well] = False

        except KeyError:
            i += 1
            continue
        i += 1
    return binary

def get_debitfun_dict(bounds=np.array([]),start=0):
    debit_functions=dict({})
    k=0
    while k<bounds.shape[0]:
        fun=debit_function()
        t=bounds[k]
        fun.t1=t[0]
        fun.t2=t[1]
        debit_functions.update({k+start:fun})
        k+=1
    return debit_functions

def set_cathegory(x,columns=np.array([])):
    cathegory=[]
    for i,c in enumerate(columns):
        if c==x:
            cathegory.append(i)
            break
    return np.array(cathegory)

def set_loc_index(index=np.array([]), R=np.array([]), T=np.array([]), D=np.array([])):
    i = 0
    res = np.empty(shape=(index.shape[0], 2))
    res.fill(np.NINF)
    while i < R.shape[1]:
        j = 0
        while j < R.shape[0]:
            ind = R[j, i]
            if ind < 0:
                break
            t2 = T[j, i]
            t1 = D[j, i]
            res[ind, 0] = t1
            res[ind, 1] = t2

            j += 1
        i += 1
    return res


#from importlib import reload
#used=np.load(path+'task\\used.npy')
#support=np.load(path+'task\\support.npy')
#group=np.load(path+'task\\groups.npy')
#Q0=np.load(path+'task\\Q0.npy')
#Q1=np.load(path+'task\\Q1.npy')
#tr=np.load(path+'task\\tr.npy')
#ts1=np.load(path+'task\\ts.npy')
#pairs=np.load(path+'pairs.npy')
#reload(op)
#index=[234,325,326]
#tr[index]=5.
#Q0[index]=10.
#Q1[index]=15.
#queue=np.load(path+'task\\queue.npy',allow_pickle=True)[()]
#support=np.empty(shape=(Q0.shape[0],2))
#support.fill(np.nan)

#service=np.load(path+'service.npy')
#equipment=np.load(path+'equipment.npy')
#wells_service=np.load(path+'wells_service.npy',allow_pickle=True)
#wells_equipment=np.load(path+'wells_equipment.npy',allow_pickle=True)

#R=np.load(path+'task\\R.npy')
#T=np.load(path+'task\\T.npy')
#D=np.load(path+'task\\D.npy')

#wsch=wells_schedule()
#wsch.t=360
#wsch.tracing=True
#wsch.fun=wsch.f14
#stop=Q0.shape[0]
#wsch.routes=R
#wsch.start=D
#wsch.end=T
#wsch.function=op.get_optim_trajectory
#epsilon=tr.mean()
#epsilon=np.inf
#wsch.fit(ts1,tr,Q0,Q1,group,support=support,used=used,stop=stop,epsilon=epsilon,service=service,equipment=equipment,wells_service=wells_service,wells_equipment=wells_equipment,prohibits=pairs)

#mask=np.isin(wsch.free,wsch.supported)
#wsch.free=wsch.free[mask]

#trace=wsch.get_routes()
#R_,T_,D_,O_=get_rout_time(trace)
#print()
#wsch.stop=group.shape[0]
#wsch.fun=wsch.f6
#wsch.routes=R
#wsch.start=D
#wsch.end=T

#Q0=np.array([1,1.5,0.5,1,1,1])
#Q1=np.array([10,5,1,np.NINF,np.NINF,np.NINF])
#tr1=np.ones(Q0.shape[0])
#ts1=np.zeros(shape=(Q0.shape[0],Q0.shape[0]))
#n=ts1.shape[0]
#for i in np.arange(ts1.shape[0]):
   # k=i+1
   # while k<ts1.shape[0]:
        #ts1[i,k]=n-k
        #ts1[k, i] = n-k
        #k+=1
#support=np.array([[1.5,2],[1,2],[1,1.5]])
#support=None
#tr1=np.array([0.1,0.1,0.5])
#groups=np.array([0,1,2,3],dtype=int)
#ts1=np.array([[0,1,1],[1,0,1],[1,1,0]])/10
#t=tr1.sum()
#wsch=wells_schedule()
#wsch.t=10
#wsch.fit(ts1,tr1,Q0,Q1,groups,support=support)
#wsch.weights=np.array([[1,2,3,np.NINF,np.NINF,np.NINF],[3,2,1,np.NINF,np.NINF,np.NINF],[np.NINF,2,1,np.NINF,np.NINF,np.NINF]])
#wsch.infindex=np.array([0,1,2,0,0,0],dtype=np.int32)
#wsch.infmask=np.array([1,1,1,0,0,0],dtype=bool)
#v=np.array([5,5,5,np.NINF,np.NINF,np.NINF])
#wsch.vector=np.array([1,1,1,0,0,0],dtype=bool)
#wsch.check_infty(v,4)
#print('')
#wsch.fun=wsch.f6
#wsch.stop=6
#trace=wsch.get_routes_v1(tracing=False)
#R,T,D=get_rout_time(trace)
#print(R)


#subpath=path+'task\\otm\\'
#otm_Q0=np.load(subpath+'Q0.npy')
#otm_Q1=np.load(subpath+'Q1.npy')
#otm_distances=np.load(subpath+'ts.npy')
#otm_group=np.load(subpath+'groups.npy')
#otm_support=np.load(subpath+'support.npy')
#service=np.load(subpath+'service.npy')
#equipment=np.load(subpath+'equipment.npy')
#otm_service=np.load(subpath+'well_service.npy',allow_pickle=True)
#otm_tr=np.load(subpath+'tr.npy')
#prohibits=np.load(subpath+'otm_binary.npy')
#expanded=np.load(subpath+'expanded.npy',allow_pickle=True)[()]
#otm_group_support=np.array([[0,24],[9,19],[0,24]])/24

#wsch=wells_schedule()
#wsch.t=360
#wsch.tracing=True
#stop=otm_Q0.shape[0]
#wsch.fun=wsch.f14
#wsch.fit(otm_distances,otm_tr,otm_Q0,otm_Q1,otm_group,support=otm_support,stop=stop,epsilon=np.inf,service=service, equipment=equipment,wells_service=otm_service,prohibits=prohibits,group_support=otm_group_support)
#wsch.update_debit_functions(expanded)
#mask=np.isin(wsch.free,wsch.supported)
#wsch.free=wsch.free[mask]
#t1=wsch.get_group_support(0.2,1)
#t2=wsch.get_group_support(0.8,1)
#trace=wsch.get_routes_v1(tracing=True)
#R,T,D,O=get_rout_time(trace)
#print('not used '+str(wsch.free.shape[0]))
#otm_service=np.load(subpath+'otm_service.npy',allow_pickle=True)
#otm_wells=np.load(subpath+'otm_wells.npy',allow_pickle=True)
#wells=pd.Index(np.load(subpath+'wells.npy',allow_pickle=True))
#wells_service=np.load(subpath+'wells_service.npy',allow_pickle=True)
#otm_prohibits=np.load(subpath+'otm_prohibits.npy',allow_pickle=True)
#binary=get_otm_binary(otm_wells,otm_service,wells,wells_service,otm_prohibits)