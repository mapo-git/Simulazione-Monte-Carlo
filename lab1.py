import numpy as np
import matplotlib.pyplot as plt

M=2000
N=1500
rng =np.random.default_rng(72631)
x=[]

def light_interference (x,N):
    ans=0.0
    x= np.linspace(-5.0, 5.0, N)
    B= 20.0
    I= 400.0
    alpha= 0.90
    d= 5.20
    E=np.sinc(alpha*x/np.pi)**2
    C=np.cos(d*x)**2
    ans= B + I*E *C
    return x, B, B+I*E, ans

x, B_, E_ , mu = light_interference (x,N)

centro = np.abs(x) < 1.2
n=rng.poisson(mu)
n_err=np.sqrt(n)
n_c = np.sum(n[centro])
mu_c = np.sum(mu[centro])
result= np.abs(n_c-mu_c)/np.sqrt(mu_c)
if result<=2:
    compatibility= "compatibile"
else:
    compatibility="non compatibile"

m=rng.poisson(mu, size=(M,N))
n_c_sim = np.sum(m[:,centro], axis=1)
media_sim = np.mean(n_c_sim)
sigma_sim = np.std(n_c_sim,ddof=1)
sigma_teo = np.sqrt(mu_c)
std = sigma_sim/np.sqrt(M)
u=np.linspace(n_c_sim.min(),n_c_sim.max(), 500)
pdf = (np.exp(-0.5 * ((u-mu_c)/sigma_teo)**2))/(sigma_teo * np.sqrt(2* np.pi))

print("Attività 1:")
print("Valori tutti finiti:", np.all(np.isfinite(mu)))
print("Valori tutti non negativi:",np.all(mu>=0))
print("Attività 2:")
print("Il confronto statistico risulta", compatibility)
print("Attività 3:")
print("La media teorica è", mu_c, " quella simulata è", media_sim)
print("La deviazione standard teorica è", sigma_teo ,"quella simulata è",sigma_sim)
print("L'errore sulla media delle simulazioni è", std)
print("Lo scarto relativo tra la media simulata e quella teorica è",100*abs(media_sim-mu_c)/mu_c, "%")

plt.figure(figsize=(10,5))
plt.plot(x,mu,"b", x, E_, "g")
plt.axhline(B_, color="red")
plt.errorbar(x,n, yerr=n_err,fmt='.')
plt.xlabel("Posizione x [mm]")
plt.ylabel("Intensità media [conteggi]")
plt.title("Modello di interferenza e diffrazione")
plt.grid()

plt.figure(figsize=(10,5))
plt.hist(n_c_sim, bins=35, density=True, alpha=0.7,color="green", label="Monte Carlo")
plt.plot(u, pdf, "b")
plt.axvline(mu_c, color="red")
plt.xlabel("Conteggi centrali")
plt.ylabel("Densità di probabilità")
plt.title("DIstribuzione dello stimatore")
plt.grid()
plt.show()