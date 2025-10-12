def check_dims(loader, name):
    xs, es = set(), set()
    for i, d in enumerate(loader.dataset):
        if hasattr(d, "x") and d.x is not None:
            xs.add(d.x.size(-1))
        if hasattr(d, "edge_attr") and d.edge_attr is not None:
            es.add(d.edge_attr.size(-1))
    print(f"[{name}] x dims: {sorted(xs)} | edge_attr dims: {sorted(es)}")
