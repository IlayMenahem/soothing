# Primary decomposition of the physical system

System: `systems/system.txt` without its last equation, the Rabinowitsch equation of kappah != 0 (28 equations, 21 variables, coefficients in Q(pi)); kappah != 0 is imposed by saturation, so the decomposed ideal is J = <P> : kappah^oo.
Computed with Alg. PrimaryBranching by `scripts/physical_primary_decomposition.py`: Alg. IdealReduce with kappah nonzero turns the 28 equations into 18 generators in 3 rounds (0.6 s), their saturation by kappah has 56 generators (0.1 s), Alg. Branching with `minAssGTZ` finds the 6 minimal primes (0.2 s), and Singular `primdecSY` decomposes the saturation (213 s).
Interreduction, factorization, the branch solver of Alg. IdealReduce (saturation by kappah), the saturation, and `primdecSY` run in the lexicographic order with the min-fill variable order of `symbolica/factor_prep.rs` (DhDphp1 > DhDphp2 > DhDphp5 > DhDphp4 > DhDphp6 > kappah > mh > DhDphp3 > DppDh1 > DppDh2 > DpDphh1 > DpDphh2 > DpDphh4 > pphh1 > pphh2 > DpDphh3 > DpDph1 > DpDph2 > e > pph > xr).
Checks with a Gröbner basis of J (0.7 s): the equations lie in J: True; kappah is a nonzerodivisor modulo J: True; the components intersect to J: True; the isolated radicals are exactly the minimal primes: True; no component is redundant: True.

Labels: R = reduced family (kappah^2 = 192 pi^2), D = doubled family (mh = 0); the number is the dimension of the slice.
Ideals are reduced Gröbner bases in the degree reverse lexicographic order (variables in name order).

| # | type | dim | degree of primary / of radical | reduced | where |
|---|---|---|---|---|---|
| 1 | isolated | 13 | 12 / 6 | no | minimal prime D13 |
| 2 | isolated | 13 | 4 / 4 | yes | minimal prime R13 |
| 3 | isolated | 12 | 18 / 9 | no | minimal prime D12 |
| 4 | isolated | 12 | 6 / 6 | yes | minimal prime R12 |
| 5 | isolated | 8 | 8 / 8 | yes | minimal prime R8 |
| 6 | isolated | 8 | 24 / 12 | no | minimal prime D8 |
| 7 | embedded | 10 | 36 / 9 | no | radical contains minimal primes D12 |
| 8 | embedded | 10 | 12 / 6 | no | radical contains minimal primes R12 |
| 9 | embedded | 8 | 20 / 10 | no | radical contains minimal primes R13 |
| 10 | embedded | 8 | 60 / 15 | no | radical contains minimal primes D13 |

## Component 1: isolated, dimension 13 (minimal prime D13)

Primary ideal:

```
2*xr-3
DppDh1+DppDh2
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-2*pph
mh^2
e^3*pph^3+(-192*pi^2)*kappah
```

Radical:

```
2*xr-3
mh
DppDh1+DppDh2
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-2*pph
e^3*pph^3+(-192*pi^2)*kappah
```

## Component 2: isolated, dimension 13 (minimal prime R13)

Primary ideal:

```
2*xr-3
DppDh1+DppDh2
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-2*pph
e*pph-kappah
kappah^2+(-192*pi^2)
```

## Component 3: isolated, dimension 12 (minimal prime D12)

Primary ideal:

```
2*xr-3
2*DpDphh3+DpDphh4-2*pphh1-4*pphh2
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-2*pph
pph^2-DpDphh4+2*pphh1
mh^2
DpDphh4*e^3*pph-2*e^3*pph*pphh1+(-192*pi^2)*kappah
DpDphh4^2*e^3-4*DpDphh4*e^3*pphh1+4*e^3*pphh1^2+(-192*pi^2)*kappah*pph
```

Radical:

```
2*xr-3
mh
2*DpDphh3+DpDphh4-2*pphh1-4*pphh2
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-2*pph
pph^2-DpDphh4+2*pphh1
DpDphh4*e^3*pph-2*e^3*pph*pphh1+(-192*pi^2)*kappah
DpDphh4^2*e^3-4*DpDphh4*e^3*pphh1+4*e^3*pphh1^2+(-192*pi^2)*kappah*pph
```

## Component 4: isolated, dimension 12 (minimal prime R12)

Primary ideal:

```
2*xr-3
2*DpDphh3+DpDphh4-2*pphh1-4*pphh2
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-2*pph
pph^2-DpDphh4+2*pphh1
e*pph-kappah
kappah^2+(-192*pi^2)
DpDphh4*e-kappah*pph-2*e*pphh1
```

## Component 5: isolated, dimension 8 (minimal prime R8)

Primary ideal:

```
2*xr-3
DppDh1
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-4*DppDh2-2*pph
DhDphp6
DhDphp4
2*DhDphp1+4*DhDphp2+2*DhDphp3+4*DhDphp5-2*DpDphh3-DpDphh4+2*pphh1+4*pphh2
2*DppDh2*pph+pph^2-4*DhDphp2-4*DhDphp5+2*DpDphh3-4*pphh2
kappah^2+(-192*pi^2)
2*DppDh2*e+e*pph-kappah
DhDphp3*e-DppDh2*kappah
4*DhDphp2*e+4*DhDphp5*e-2*DpDphh3*e-kappah*pph+4*e*pphh2
4*DppDh2^2-pph^2+4*DhDphp2-2*DhDphp3+4*DhDphp5-2*DpDphh3+4*pphh2
4*DhDphp2*DppDh2+4*DhDphp5*DppDh2-2*DpDphh3*DppDh2-DhDphp3*pph+4*DppDh2*pphh2
2*DhDphp2*pph^2+DhDphp3*pph^2+2*DhDphp5*pph^2-DpDphh3*pph^2+2*pph^2*pphh2-8*DhDphp2^2-16*DhDphp2*DhDphp5-8*DhDphp5^2+8*DhDphp2*DpDphh3+8*DhDphp5*DpDphh3-2*DpDphh3^2-16*DhDphp2*pphh2-16*DhDphp5*pphh2+8*DpDphh3*pphh2-8*pphh2^2
```

## Component 6: isolated, dimension 8 (minimal prime D8)

Primary ideal:

```
2*xr-3
DppDh1
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-4*DppDh2-2*pph
DhDphp6
DhDphp4
2*DhDphp1+4*DhDphp2+2*DhDphp3+4*DhDphp5-2*DpDphh3-DpDphh4+2*pphh1+4*pphh2
2*DppDh2*pph+pph^2-4*DhDphp2-4*DhDphp5+2*DpDphh3-4*pphh2
mh^2
4*DppDh2^2-pph^2+4*DhDphp2-2*DhDphp3+4*DhDphp5-2*DpDphh3+4*pphh2
4*DhDphp2*DppDh2+4*DhDphp5*DppDh2-2*DpDphh3*DppDh2-DhDphp3*pph+4*DppDh2*pphh2
2*DhDphp2*pph^2+DhDphp3*pph^2+2*DhDphp5*pph^2-DpDphh3*pph^2+2*pph^2*pphh2-8*DhDphp2^2-16*DhDphp2*DhDphp5-8*DhDphp5^2+8*DhDphp2*DpDphh3+8*DhDphp5*DpDphh3-2*DpDphh3^2-16*DhDphp2*pphh2-16*DhDphp5*pphh2+8*DpDphh3*pphh2-8*pphh2^2
2*DhDphp3*DppDh2*e^3+2*DhDphp2*e^3*pph+2*DhDphp3*e^3*pph+2*DhDphp5*e^3*pph-DpDphh3*e^3*pph+2*e^3*pph*pphh2+(-96*pi^2)*kappah
2*DhDphp2*DhDphp3*e^3+DhDphp3^2*e^3+2*DhDphp3*DhDphp5*e^3-DhDphp3*DpDphh3*e^3+2*DhDphp3*e^3*pphh2+(-96*pi^2)*DppDh2*kappah
4*DhDphp2^2*e^3-DhDphp3^2*e^3+8*DhDphp2*DhDphp5*e^3+4*DhDphp5^2*e^3-4*DhDphp2*DpDphh3*e^3-4*DhDphp5*DpDphh3*e^3+DpDphh3^2*e^3+8*DhDphp2*e^3*pphh2+8*DhDphp5*e^3*pphh2-4*DpDphh3*e^3*pphh2+4*e^3*pphh2^2+(96*pi^2)*DppDh2*kappah+(-48*pi^2)*kappah*pph
DhDphp3^3*e^3+(24*pi^2)*kappah*pph^3+(-96*pi^2)*DhDphp3*DppDh2*kappah+(-96*pi^2)*DhDphp2*kappah*pph+(48*pi^2)*DhDphp3*kappah*pph+(-96*pi^2)*DhDphp5*kappah*pph+(48*pi^2)*DpDphh3*kappah*pph+(-96*pi^2)*kappah*pph*pphh2
```

Radical:

```
2*xr-3
mh
DppDh1
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-4*DppDh2-2*pph
DhDphp6
DhDphp4
2*DhDphp1+4*DhDphp2+2*DhDphp3+4*DhDphp5-2*DpDphh3-DpDphh4+2*pphh1+4*pphh2
2*DppDh2*pph+pph^2-4*DhDphp2-4*DhDphp5+2*DpDphh3-4*pphh2
4*DppDh2^2-pph^2+4*DhDphp2-2*DhDphp3+4*DhDphp5-2*DpDphh3+4*pphh2
4*DhDphp2*DppDh2+4*DhDphp5*DppDh2-2*DpDphh3*DppDh2-DhDphp3*pph+4*DppDh2*pphh2
2*DhDphp2*pph^2+DhDphp3*pph^2+2*DhDphp5*pph^2-DpDphh3*pph^2+2*pph^2*pphh2-8*DhDphp2^2-16*DhDphp2*DhDphp5-8*DhDphp5^2+8*DhDphp2*DpDphh3+8*DhDphp5*DpDphh3-2*DpDphh3^2-16*DhDphp2*pphh2-16*DhDphp5*pphh2+8*DpDphh3*pphh2-8*pphh2^2
2*DhDphp3*DppDh2*e^3+2*DhDphp2*e^3*pph+2*DhDphp3*e^3*pph+2*DhDphp5*e^3*pph-DpDphh3*e^3*pph+2*e^3*pph*pphh2+(-96*pi^2)*kappah
2*DhDphp2*DhDphp3*e^3+DhDphp3^2*e^3+2*DhDphp3*DhDphp5*e^3-DhDphp3*DpDphh3*e^3+2*DhDphp3*e^3*pphh2+(-96*pi^2)*DppDh2*kappah
4*DhDphp2^2*e^3-DhDphp3^2*e^3+8*DhDphp2*DhDphp5*e^3+4*DhDphp5^2*e^3-4*DhDphp2*DpDphh3*e^3-4*DhDphp5*DpDphh3*e^3+DpDphh3^2*e^3+8*DhDphp2*e^3*pphh2+8*DhDphp5*e^3*pphh2-4*DpDphh3*e^3*pphh2+4*e^3*pphh2^2+(96*pi^2)*DppDh2*kappah+(-48*pi^2)*kappah*pph
DhDphp3^3*e^3+(24*pi^2)*kappah*pph^3+(-96*pi^2)*DhDphp3*DppDh2*kappah+(-96*pi^2)*DhDphp2*kappah*pph+(48*pi^2)*DhDphp3*kappah*pph+(-96*pi^2)*DhDphp5*kappah*pph+(48*pi^2)*DpDphh3*kappah*pph+(-96*pi^2)*kappah*pph*pphh2
```

## Component 7: embedded, dimension 10 (radical contains minimal primes D12)

Primary ideal:

```
2*xr-3
DpDphh4+5*pphh1-2*pphh2
DpDphh3-4*pphh1
8*DpDphh2+pphh1+6*pphh2
8*DpDphh1+7*pphh1+2*pphh2
DpDph2+pph
DpDph1-2*pph
pphh1^2-4*pphh1*pphh2+4*pphh2^2
pph^2+6*pphh1
mh^2*pphh1-2*mh^2*pphh2
mh^4
3*e^3*pphh2^2+(32*pi^2)*kappah*mh^2*pph+(384*pi^2)*e*mh^2*pphh2+(-4*pi^2)*kappah*pph
3*e^3*pphh1*pphh2+(32*pi^2)*kappah*mh^2*pph+(384*pi^2)*e*mh^2*pphh2+(-8*pi^2)*kappah*pph
8*kappah*mh^2*pph*pphh2+96*e*mh^2*pphh2^2-kappah*pph*pphh1+2*kappah*pph*pphh2
8*e*mh^2*pph*pphh2-8*kappah*mh^2*pphh2+kappah*pphh1-2*kappah*pphh2
e^3*pph*pphh2+(64*pi^2)*e*mh^2*pph+(-64*pi^2)*kappah*mh^2+(16*pi^2)*kappah
8*kappah^2*mh^2*pphh2-e*kappah*pph*pphh1+2*e*kappah*pph*pphh2+12*e^2*pphh1*pphh2-24*e^2*pphh2^2-kappah^2*pphh1+2*kappah^2*pphh2+(-1536*pi^2)*mh^2*pphh2
8*e*kappah*mh^2*pphh2-e^2*pph*pphh1+2*e^2*pph*pphh2+(128*pi^2)*mh^2*pph-e*kappah*pphh1+2*e*kappah*pphh2
8*e^2*mh^2*pphh2-e^2*pphh1+2*e^2*pphh2+(128*pi^2)*mh^2
e^3*pph*pphh1+(32*pi^2)*kappah
```

Radical:

```
2*xr-3
pphh1-2*pphh2
mh
DpDphh4+8*pphh2
DpDphh3-8*pphh2
DpDphh2+pphh2
DpDphh1+2*pphh2
DpDph2+pph
DpDph1-2*pph
pph^2+12*pphh2
3*e^3*pphh2^2+(-4*pi^2)*kappah*pph
e^3*pph*pphh2+(16*pi^2)*kappah
```

## Component 8: embedded, dimension 10 (radical contains minimal primes R12)

Primary ideal:

```
2*xr-3
DpDphh3-4*pphh1
8*DpDphh2+DpDphh4+6*pphh1+4*pphh2
8*DpDphh1-DpDphh4+2*pphh1+4*pphh2
DpDph2+pph
DpDph1-2*pph
pphh1^2-4*pphh1*pphh2+4*pphh2^2
DpDphh4*pphh1-2*DpDphh4*pphh2+8*pphh1*pphh2-16*pphh2^2
pph^2-2*DpDphh4-4*pphh1+4*pphh2
4*DpDphh4*e-kappah*pph+14*e*pphh1-8*e*pphh2
DpDphh4^2+16*DpDphh4*pphh2+64*pphh2^2
6*e*pph*pphh2-DpDphh4*kappah-5*kappah*pphh1-4*kappah*pphh2
kappah^2*pphh2+(96*pi^2)*DpDphh4+(480*pi^2)*pphh1+(-384*pi^2)*pphh2
kappah*pph*pphh1-2*kappah*pph*pphh2+12*e*pphh1*pphh2-24*e*pphh2^2
3*e*pph*pphh1-DpDphh4*kappah-8*kappah*pphh1+2*kappah*pphh2
kappah^2*pphh1+(192*pi^2)*DpDphh4+(768*pi^2)*pphh1+(-384*pi^2)*pphh2
e^2*pphh1+(32*pi^2)
kappah^2*pph+18*e*kappah*pphh1+(384*pi^2)*pph
3*e*kappah*pph-2*kappah^2+(-192*pi^2)
DpDphh4*kappah*pph+11*kappah*pph*pphh2-42*e*pphh1*pphh2+120*e*pphh2^2
4*DpDphh4*mh^2+20*mh^2*pphh1-8*mh^2*pphh2+3*DpDphh4+18*pphh1-12*pphh2
kappah^3+(576*pi^2)*e*pph+(-768*pi^2)*kappah
DpDphh4*kappah^2+(-960*pi^2)*DpDphh4+(-3840*pi^2)*pphh1+(1536*pi^2)*pphh2
3*e*kappah*pphh2^2+(-8*pi^2)*DpDphh4*pph+(-64*pi^2)*pph*pphh1+(112*pi^2)*pph*pphh2
e^2*pphh2^2+(-8*pi^2)*pphh1+(32*pi^2)*pphh2
3*e*kappah*pphh1*pphh2+(-16*pi^2)*DpDphh4*pph+(-80*pi^2)*pph*pphh1+(128*pi^2)*pph*pphh2
4*kappah*mh^2*pph+24*e*mh^2*pphh1+3*kappah*pph+30*e*pphh1-24*e*pphh2
(64*pi^2)*e*mh^2*pph+e^2*kappah*pphh2+(-64*pi^2)*kappah*mh^2+(48*pi^2)*e*pph+(-32*pi^2)*kappah
4*kappah^2*mh^2-36*e^2*pphh2+3*kappah^2+(-768*pi^2)*mh^2+(-1152*pi^2)
2*e*kappah*mh^2*pphh1+(64*pi^2)*mh^2*pph+e*kappah*pphh1+e*kappah*pphh2+(48*pi^2)*pph
```

Radical:

```
2*xr-3
pphh1-2*pphh2
DpDphh4+8*pphh2
DpDphh3-8*pphh2
DpDphh2+pphh2
DpDphh1+2*pphh2
DpDph2+pph
DpDph1-2*pph
pph^2+12*pphh2
kappah*pph+12*e*pphh2
e*pph-kappah
kappah^2+(-192*pi^2)
e*kappah*pphh2+(16*pi^2)*pph
e^2*pphh2+(16*pi^2)
```

## Component 9: embedded, dimension 8 (radical contains minimal primes R13)

Primary ideal:

```
2*xr-3
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-4*DppDh2-2*pph
DhDphp6
DhDphp4
DppDh2*pph-DhDphp3
kappah^2+(-192*pi^2)
2*DppDh2*e+e*pph-kappah
DhDphp3*e-DppDh2*kappah
DppDh2^2
DppDh1*DppDh2
DhDphp3*DppDh2
DppDh1^2
2*DpDphh3*DppDh1+DpDphh4*DppDh1+2*DhDphp1*DppDh2+4*DhDphp2*DppDh2+4*DhDphp5*DppDh2-2*DpDphh3*DppDh2-DpDphh4*DppDh2-2*DppDh1*pphh1+2*DppDh2*pphh1-4*DppDh1*pphh2+4*DppDh2*pphh2
DhDphp3*DppDh1
DhDphp3^2
e*pph^2+2*DppDh2*kappah-kappah*pph
DppDh1*pph^2-DpDphh4*DppDh1-2*DhDphp1*DppDh2+DpDphh4*DppDh2-DhDphp3*pph+2*DppDh1*pphh1-2*DppDh2*pphh1
DhDphp1*pph^2+2*DhDphp2*pph^2+2*DhDphp5*pph^2+4*DhDphp1*DhDphp3+8*DhDphp2*DhDphp3+8*DhDphp3*DhDphp5+2*DhDphp1*DpDphh3-2*DhDphp2*DpDphh4-2*DhDphp5*DpDphh4+4*DhDphp2*pphh1+4*DhDphp5*pphh1-4*DhDphp1*pphh2
DppDh1*e*pph-DppDh1*kappah
2*DpDphh4*DppDh1*e-2*DhDphp1*e*pph+DpDphh4*e*pph-2*DppDh1*kappah*pph-4*DppDh1*e*pphh1-2*e*pph*pphh1+2*DhDphp1*kappah+2*DhDphp3*kappah-DpDphh4*kappah+2*kappah*pphh1
2*DhDphp1*DpDphh3*e-2*DhDphp2*DpDphh4*e-2*DhDphp5*DpDphh4*e+2*DhDphp1*DppDh2*kappah+4*DhDphp2*DppDh2*kappah+4*DhDphp5*DppDh2*kappah+DhDphp1*kappah*pph+2*DhDphp2*kappah*pph+2*DhDphp5*kappah*pph+4*DhDphp2*e*pphh1+4*DhDphp5*e*pphh1-4*DhDphp1*e*pphh2
2*DhDphp1*DpDphh3*DppDh2-2*DhDphp2*DpDphh4*DppDh2-2*DhDphp5*DpDphh4*DppDh2+DhDphp1*DhDphp3*pph+2*DhDphp2*DhDphp3*pph+2*DhDphp3*DhDphp5*pph+4*DhDphp2*DppDh2*pphh1+4*DhDphp5*DppDh2*pphh1-4*DhDphp1*DppDh2*pphh2
```

Radical:

```
2*xr-3
DppDh2
DppDh1
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-2*pph
DhDphp6
DhDphp4
DhDphp3
e*pph-kappah
kappah^2+(-192*pi^2)
DhDphp1*pph^2+2*DhDphp2*pph^2+2*DhDphp5*pph^2+2*DhDphp1*DpDphh3-2*DhDphp2*DpDphh4-2*DhDphp5*DpDphh4+4*DhDphp2*pphh1+4*DhDphp5*pphh1-4*DhDphp1*pphh2
2*DhDphp1*DpDphh3*e-2*DhDphp2*DpDphh4*e-2*DhDphp5*DpDphh4*e+DhDphp1*kappah*pph+2*DhDphp2*kappah*pph+2*DhDphp5*kappah*pph+4*DhDphp2*e*pphh1+4*DhDphp5*e*pphh1-4*DhDphp1*e*pphh2
```

## Component 10: embedded, dimension 8 (radical contains minimal primes D13)

Primary ideal:

```
2*xr-3
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-4*DppDh2-2*pph
DhDphp6
DhDphp4
DhDphp3
mh^2
DppDh2^2
DppDh1*DppDh2
DppDh1^2
2*DpDphh3*DppDh1+DpDphh4*DppDh1+2*DhDphp1*DppDh2+4*DhDphp2*DppDh2+4*DhDphp5*DppDh2-2*DpDphh3*DppDh2-DpDphh4*DppDh2-2*DppDh1*pphh1+2*DppDh2*pphh1-4*DppDh1*pphh2+4*DppDh2*pphh2
DppDh1*pph^2-DppDh2*pph^2-DpDphh4*DppDh1-2*DhDphp1*DppDh2+DpDphh4*DppDh2+2*DppDh1*pphh1-2*DppDh2*pphh1
DhDphp1*pph^2+2*DhDphp2*pph^2+2*DhDphp5*pph^2+2*DhDphp1*DpDphh3-2*DhDphp2*DpDphh4-2*DhDphp5*DpDphh4+4*DhDphp2*pphh1+4*DhDphp5*pphh1-4*DhDphp1*pphh2
6*DppDh2*e^3*pph^2+e^3*pph^3+(-192*pi^2)*kappah
DpDphh4*DppDh1*e^3*pph+2*DhDphp1*DppDh2*e^3*pph-DpDphh4*DppDh2*e^3*pph-2*DppDh1*e^3*pph*pphh1+2*DppDh2*e^3*pph*pphh1+(-192*pi^2)*DppDh1*kappah+(192*pi^2)*DppDh2*kappah
6*DhDphp1*DpDphh3*DppDh2*e^3-6*DhDphp2*DpDphh4*DppDh2*e^3-6*DhDphp5*DpDphh4*DppDh2*e^3+DhDphp1*DpDphh3*e^3*pph-DhDphp2*DpDphh4*e^3*pph-DhDphp5*DpDphh4*e^3*pph+12*DhDphp2*DppDh2*e^3*pphh1+12*DhDphp5*DppDh2*e^3*pphh1+2*DhDphp2*e^3*pph*pphh1+2*DhDphp5*e^3*pph*pphh1-12*DhDphp1*DppDh2*e^3*pphh2-2*DhDphp1*e^3*pph*pphh2+(96*pi^2)*DhDphp1*kappah+(192*pi^2)*DhDphp2*kappah+(192*pi^2)*DhDphp5*kappah
e^3*pph^4+(1152*pi^2)*DppDh2*kappah+(-192*pi^2)*kappah*pph
2*DhDphp2*e^3*pph^3+2*DhDphp5*e^3*pph^3+3*DpDphh4^2*DppDh1*e^3+6*DhDphp1*DpDphh4*DppDh2*e^3-3*DpDphh4^2*DppDh2*e^3+2*DhDphp1*DpDphh3*e^3*pph-2*DhDphp2*DpDphh4*e^3*pph-2*DhDphp5*DpDphh4*e^3*pph-12*DpDphh4*DppDh1*e^3*pphh1-12*DhDphp1*DppDh2*e^3*pphh1+12*DpDphh4*DppDh2*e^3*pphh1+4*DhDphp2*e^3*pph*pphh1+4*DhDphp5*e^3*pph*pphh1+12*DppDh1*e^3*pphh1^2-12*DppDh2*e^3*pphh1^2-4*DhDphp1*e^3*pph*pphh2+(-576*pi^2)*DppDh1*kappah*pph+(576*pi^2)*DppDh2*kappah*pph+(192*pi^2)*DhDphp1*kappah
2*DhDphp2*DpDphh3*e^3*pph^2+2*DhDphp5*DpDphh3*e^3*pph^2+DhDphp2*DpDphh4*e^3*pph^2+DhDphp5*DpDphh4*e^3*pph^2-2*DhDphp2*e^3*pph^2*pphh1-2*DhDphp5*e^3*pph^2*pphh1-4*DhDphp2*e^3*pph^2*pphh2-4*DhDphp5*e^3*pph^2*pphh2+2*DhDphp1*DpDphh3^2*e^3-2*DhDphp2*DpDphh3*DpDphh4*e^3-2*DhDphp5*DpDphh3*DpDphh4*e^3+4*DhDphp2*DpDphh3*e^3*pphh1+4*DhDphp5*DpDphh3*e^3*pphh1-8*DhDphp1*DpDphh3*e^3*pphh2+4*DhDphp2*DpDphh4*e^3*pphh2+4*DhDphp5*DpDphh4*e^3*pphh2-8*DhDphp2*e^3*pphh1*pphh2-8*DhDphp5*e^3*pphh1*pphh2+8*DhDphp1*e^3*pphh2^2+(576*pi^2)*DhDphp1*DppDh2*kappah+(1152*pi^2)*DhDphp2*DppDh2*kappah+(1152*pi^2)*DhDphp5*DppDh2*kappah+(-96*pi^2)*DhDphp1*kappah*pph+(-192*pi^2)*DhDphp2*kappah*pph+(-192*pi^2)*DhDphp5*kappah*pph
3*DhDphp1*DpDphh4^2*DppDh1*e^3+6*DhDphp2*DpDphh4^2*DppDh1*e^3+6*DhDphp5*DpDphh4^2*DppDh1*e^3+6*DhDphp1^2*DpDphh4*DppDh2*e^3+12*DhDphp1*DhDphp2*DpDphh4*DppDh2*e^3+12*DhDphp1*DhDphp5*DpDphh4*DppDh2*e^3-3*DhDphp1*DpDphh4^2*DppDh2*e^3-6*DhDphp2*DpDphh4^2*DppDh2*e^3-6*DhDphp5*DpDphh4^2*DppDh2*e^3+2*DhDphp1^2*DpDphh3*e^3*pph-2*DhDphp1*DhDphp2*DpDphh4*e^3*pph-2*DhDphp1*DhDphp5*DpDphh4*e^3*pph-12*DhDphp1*DpDphh4*DppDh1*e^3*pphh1-24*DhDphp2*DpDphh4*DppDh1*e^3*pphh1-24*DhDphp5*DpDphh4*DppDh1*e^3*pphh1-12*DhDphp1^2*DppDh2*e^3*pphh1-24*DhDphp1*DhDphp2*DppDh2*e^3*pphh1-24*DhDphp1*DhDphp5*DppDh2*e^3*pphh1+12*DhDphp1*DpDphh4*DppDh2*e^3*pphh1+24*DhDphp2*DpDphh4*DppDh2*e^3*pphh1+24*DhDphp5*DpDphh4*DppDh2*e^3*pphh1+4*DhDphp1*DhDphp2*e^3*pph*pphh1+4*DhDphp1*DhDphp5*e^3*pph*pphh1+12*DhDphp1*DppDh1*e^3*pphh1^2+24*DhDphp2*DppDh1*e^3*pphh1^2+24*DhDphp5*DppDh1*e^3*pphh1^2-12*DhDphp1*DppDh2*e^3*pphh1^2-24*DhDphp2*DppDh2*e^3*pphh1^2-24*DhDphp5*DppDh2*e^3*pphh1^2-4*DhDphp1^2*e^3*pph*pphh2+(-576*pi^2)*DhDphp1*DppDh1*kappah*pph+(-1152*pi^2)*DhDphp2*DppDh1*kappah*pph+(-1152*pi^2)*DhDphp5*DppDh1*kappah*pph+(576*pi^2)*DhDphp1*DppDh2*kappah*pph+(1152*pi^2)*DhDphp2*DppDh2*kappah*pph+(1152*pi^2)*DhDphp5*DppDh2*kappah*pph+(192*pi^2)*DhDphp1^2*kappah+(384*pi^2)*DhDphp1*DhDphp2*kappah+(384*pi^2)*DhDphp1*DhDphp5*kappah
DhDphp1^2*DpDphh3^2*e^3-2*DhDphp1*DhDphp2*DpDphh3*DpDphh4*e^3-2*DhDphp1*DhDphp5*DpDphh3*DpDphh4*e^3+DhDphp2^2*DpDphh4^2*e^3+2*DhDphp2*DhDphp5*DpDphh4^2*e^3+DhDphp5^2*DpDphh4^2*e^3+4*DhDphp1*DhDphp2*DpDphh3*e^3*pphh1+4*DhDphp1*DhDphp5*DpDphh3*e^3*pphh1-4*DhDphp2^2*DpDphh4*e^3*pphh1-8*DhDphp2*DhDphp5*DpDphh4*e^3*pphh1-4*DhDphp5^2*DpDphh4*e^3*pphh1+4*DhDphp2^2*e^3*pphh1^2+8*DhDphp2*DhDphp5*e^3*pphh1^2+4*DhDphp5^2*e^3*pphh1^2-4*DhDphp1^2*DpDphh3*e^3*pphh2+4*DhDphp1*DhDphp2*DpDphh4*e^3*pphh2+4*DhDphp1*DhDphp5*DpDphh4*e^3*pphh2-8*DhDphp1*DhDphp2*e^3*pphh1*pphh2-8*DhDphp1*DhDphp5*e^3*pphh1*pphh2+4*DhDphp1^2*e^3*pphh2^2+(288*pi^2)*DhDphp1^2*DppDh2*kappah+(1152*pi^2)*DhDphp1*DhDphp2*DppDh2*kappah+(1152*pi^2)*DhDphp2^2*DppDh2*kappah+(1152*pi^2)*DhDphp1*DhDphp5*DppDh2*kappah+(2304*pi^2)*DhDphp2*DhDphp5*DppDh2*kappah+(1152*pi^2)*DhDphp5^2*DppDh2*kappah+(-48*pi^2)*DhDphp1^2*kappah*pph+(-192*pi^2)*DhDphp1*DhDphp2*kappah*pph+(-192*pi^2)*DhDphp2^2*kappah*pph+(-192*pi^2)*DhDphp1*DhDphp5*kappah*pph+(-384*pi^2)*DhDphp2*DhDphp5*kappah*pph+(-192*pi^2)*DhDphp5^2*kappah*pph
```

Radical:

```
2*xr-3
mh
DppDh2
DppDh1
DpDphh2+pphh2
DpDphh1+pphh1
DpDph2+pph
DpDph1-2*pph
DhDphp6
DhDphp4
DhDphp3
DhDphp1*pph^2+2*DhDphp2*pph^2+2*DhDphp5*pph^2+2*DhDphp1*DpDphh3-2*DhDphp2*DpDphh4-2*DhDphp5*DpDphh4+4*DhDphp2*pphh1+4*DhDphp5*pphh1-4*DhDphp1*pphh2
e^3*pph^3+(-192*pi^2)*kappah
DhDphp1*DpDphh3*e^3*pph-DhDphp2*DpDphh4*e^3*pph-DhDphp5*DpDphh4*e^3*pph+2*DhDphp2*e^3*pph*pphh1+2*DhDphp5*e^3*pph*pphh1-2*DhDphp1*e^3*pph*pphh2+(96*pi^2)*DhDphp1*kappah+(192*pi^2)*DhDphp2*kappah+(192*pi^2)*DhDphp5*kappah
2*DhDphp2*DpDphh3*e^3*pph^2+2*DhDphp5*DpDphh3*e^3*pph^2+DhDphp2*DpDphh4*e^3*pph^2+DhDphp5*DpDphh4*e^3*pph^2-2*DhDphp2*e^3*pph^2*pphh1-2*DhDphp5*e^3*pph^2*pphh1-4*DhDphp2*e^3*pph^2*pphh2-4*DhDphp5*e^3*pph^2*pphh2+2*DhDphp1*DpDphh3^2*e^3-2*DhDphp2*DpDphh3*DpDphh4*e^3-2*DhDphp5*DpDphh3*DpDphh4*e^3+4*DhDphp2*DpDphh3*e^3*pphh1+4*DhDphp5*DpDphh3*e^3*pphh1-8*DhDphp1*DpDphh3*e^3*pphh2+4*DhDphp2*DpDphh4*e^3*pphh2+4*DhDphp5*DpDphh4*e^3*pphh2-8*DhDphp2*e^3*pphh1*pphh2-8*DhDphp5*e^3*pphh1*pphh2+8*DhDphp1*e^3*pphh2^2+(-96*pi^2)*DhDphp1*kappah*pph+(-192*pi^2)*DhDphp2*kappah*pph+(-192*pi^2)*DhDphp5*kappah*pph
DhDphp1^2*DpDphh3^2*e^3-2*DhDphp1*DhDphp2*DpDphh3*DpDphh4*e^3-2*DhDphp1*DhDphp5*DpDphh3*DpDphh4*e^3+DhDphp2^2*DpDphh4^2*e^3+2*DhDphp2*DhDphp5*DpDphh4^2*e^3+DhDphp5^2*DpDphh4^2*e^3+4*DhDphp1*DhDphp2*DpDphh3*e^3*pphh1+4*DhDphp1*DhDphp5*DpDphh3*e^3*pphh1-4*DhDphp2^2*DpDphh4*e^3*pphh1-8*DhDphp2*DhDphp5*DpDphh4*e^3*pphh1-4*DhDphp5^2*DpDphh4*e^3*pphh1+4*DhDphp2^2*e^3*pphh1^2+8*DhDphp2*DhDphp5*e^3*pphh1^2+4*DhDphp5^2*e^3*pphh1^2-4*DhDphp1^2*DpDphh3*e^3*pphh2+4*DhDphp1*DhDphp2*DpDphh4*e^3*pphh2+4*DhDphp1*DhDphp5*DpDphh4*e^3*pphh2-8*DhDphp1*DhDphp2*e^3*pphh1*pphh2-8*DhDphp1*DhDphp5*e^3*pphh1*pphh2+4*DhDphp1^2*e^3*pphh2^2+(-48*pi^2)*DhDphp1^2*kappah*pph+(-192*pi^2)*DhDphp1*DhDphp2*kappah*pph+(-192*pi^2)*DhDphp2^2*kappah*pph+(-192*pi^2)*DhDphp1*DhDphp5*kappah*pph+(-384*pi^2)*DhDphp2*DhDphp5*kappah*pph+(-192*pi^2)*DhDphp5^2*kappah*pph
```
