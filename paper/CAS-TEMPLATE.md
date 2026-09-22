# Elsevier CAS template

`paper.tex` uses the single-column `cas-sc` template from the exact publisher archive:

https://assets.ctfassets.net/o78em1y1w4i4/5uFmLZJTPDMAUjFnHRpjj8/6f19a979146eb93263763d87a894ab0d/els-cas-templates.zip

The bundled `cas-sc.cls`, `cas-common.sty`, and `cas-model2-names.bst` are copied unchanged from that archive (CAS version 2.4, 2024-05-04). Their original license notices are retained.

Build from the repository root with:

```sh
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/paper.tex
```

CAS captures the abstract verbatim, so its text is kept directly in `paper.tex`.
