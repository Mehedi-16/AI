
print("P\tQ\tP->Q\tModus Ponens (P and P->Q implies Q)")

for P in [True, False]:
    for Q in [True, False]:
        P_implies_Q = (not P) or Q  
        modus_ponens_valid = P and P_implies_Q and Q
        print(f"{P}\t{Q}\t{P_implies_Q}\t{modus_ponens_valid}")
