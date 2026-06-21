# Archived Graph-JEPA explorations

Graph-JEPA was explored as a separate, masked-latent anomaly-scoring direction.
It is not part of the controlled SIGReg/PEIRA ambient-versus-SPD-tangent result.

- [`v1/`](v1/) is the first standalone attempt. It is retained for provenance;
  its original data helpers were incomplete, so it is not a supported pipeline.
- [`v2/`](v2/) is the later TCP-Graph-JEPA experiment with its own model,
  scripts, configuration, results, and targeted tests.

Run its targeted tests only when working on the archive:

```bash
python -m pytest archive/graph_jepa/v2/tests -q
```

The main project and its default test suite deliberately do not import this
archive.
