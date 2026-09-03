# Release checklist

- [ ] Remove absolute local/server paths.
- [ ] Remove usernames, SSH hosts, passwords, API keys, and tokens.
- [ ] Do not commit raw WSIs, model checkpoints, or embeddings.
- [ ] Include the audited 71-case manifest.
- [ ] Include same-center coordinate/anchor manifests.
- [ ] Document model IDs/checkpoint versions and preprocessing.
- [ ] Document UNI audit: official checkpoint, 224×224 input, 1024-d output, no NaN/Inf.
- [ ] Document all random seeds and statistical units.
- [ ] Verify released scripts reproduce Figures 2–8.
- [ ] For blinded review, do not expose author identity in a public repository.
