import train

for seed in [1, 7, 42, 123]:
    train.SEED = seed
    m = train.train()
    tc, tt = train.evaluate_argmax(m, train.TEST_TRIPLES, label="TEST", verbose=False)
    vc, vt = train.evaluate_argmax(m, train.VAL_TRIPLES, label="VAL", verbose=False)
    print(f"seed={seed}: TEST_TRIPLES {tc}/{tt}, VAL_TRIPLES {vc}/{vt}", flush=True)
