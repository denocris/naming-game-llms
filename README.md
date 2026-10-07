# Microscopic dynamics of consensus formation in multi-agent LLM Naming Games

Code and data for the paper *Microscopic dynamics of consensus formation in multi-agent LLM Naming Games*.

[![arXiv](https://img.shields.io/badge/arXiv-2608.02178-b31b1b.svg)](https://arxiv.org/abs/2608.02178)
<!-- After the Zenodo release, add: [![DOI](https://zenodo.org/badge/DOI/<DOI>.svg)](https://doi.org/<DOI>) -->

## What the paper does

The Naming Game (NG) is the minimal model of how a population bootstraps a shared convention through pairwise negotiation. We keep its topology and update rule but replace the listener's deterministic inventory check with a single-token call to a Large Language Model (LLM) at decoding temperature $T$. Each interaction is then fully characterized by two conditional acceptance rates,

$$
\pi(T) = P(\text{YES} \mid w \in P_j), \qquad \phi(T) = P(\text{YES} \mid w \notin P_j).
$$

The deterministic NG is $(\pi, \phi) = (1, 0)$, and the stochastic negotiation model of Baronchelli, Dall'Asta, Barrat and Loreto ([Phys. Rev. E 76, 051102](https://doi.org/10.1103/PhysRevE.76.051102)) is the $\phi = 0$ edge. In our setting both rates are emergent properties of the LLM rather than imposed parameters.

## Main results

1. A complete-graph mean-field theory of the two-rate dynamics. Its two-word sector gives the instability condition $3\pi - 2\phi - 1 > 0$, which reduces to the known threshold $\beta_c = 1/3$ at $\phi = 0$ and defines a critical line in the $(\pi, \phi)$ plane. The exact mean-field hierarchy for an arbitrary vocabulary and an inventory-size closure are given in an appendix.
2. Across three open-weight architectures, the measured $(\pi, \phi)$ place the listeners in three distinct regimes (permissive, near-deterministic, conservative), each with its own macroscopic signature, including an inverted temperature ordering of the consensus time in the conservative case.
3. Finite-size scaling and temperature-response measurements for $N$ between 50 and 150. The effective exponent $\beta(T)$ in $t_{\rm conv} \sim N^{\beta}$ and the temperature sensitivity $\alpha$ in $t_c \sim e^{\alpha T}$ are architecture-dependent. We state explicitly that these are effective exponents over a limited size range and not asymptotic ones.

## Authors

- **Cristiano De Nobili**, Critiqality, Milan, Italy ([ORCID](https://orcid.org/0000-0002-8429-1831), cristiano@critiqality.ai)
- **Vijayasri Iyer**, Independent Researcher
- **Alessandro Codello**, DSMN, Ca' Foscari University of Venice, Italy; IFFI, Universidad de la República, Montevideo, Uruguay
- **Raffaella Burioni**, Dipartimento di Scienze Matematiche, Fisiche e Informatiche, Università degli Studi di Parma, Italy; INFN, Gruppo Collegato di Parma, Italy

## Useful links

- Paper (arXiv): https://arxiv.org/abs/2608.02178
- Blog post: https://cristianodenobili.com/blog/blogs/how-consensus-emerge-in-multi-agent-llm-systems

## Simulation setup

These settings are those used in the paper.

| Item | Value |
|---|---|
| Models (served locally with [Ollama](https://ollama.com)) | `llama3.1:8b`, `mistral:7b`, `phi3:14b` |
| Interaction | Complete graph, ordered speaker–listener pair drawn uniformly at each step |
| Listener decision | Single YES/NO token at temperature $T$ |
| Temperatures | $T \in \{0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0\}$ |
| Population size | $N = 150$ for the main runs; $N \in \{50, 70, 90, 110, 150\}$ for scaling |
| Horizon | Up to $10^5$ steps ($1.75 \times 10^5$ for `phi3:14b`) |
| Seeds | 10 to 15 per configuration |
| Vocabulary | $\lvert\mathcal{V}\rvert = 10^4$ English words |
| Consensus | $N_d(t_c) = 1$, where $N_d$ is the number of distinct words |

Prompt used by the listener:

- **System:** *You are an agent with your own language and vocabulary. You can and must reply with yes or no.*
- **User:** *Your words are: $P_j$. Do we add $w$ to the list?*

## Requirements

- Python and the packages in `[requirements.txt / environment.yml]`
- [Ollama](https://ollama.com), with the models pulled:

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull phi3:14b
```

Model tags do not pin the weights. The digests used for the paper are listed in `[FILE]`.

## Reproducing the results

```bash
nohup python3 pipeline.py --config config-ollama.yaml > nohup_ollama.log &
```

## Data

The data supporting the results are archived on Zenodo: [DOI](https://doi.org/10.5281/zenodo.23207716). They include the per-step logs from which $\pi(t)$, $\phi(t)$, $N_d(t)$, $\bar{k}(t)$ and the consensus times were computed.

## Citation

If you use this code or data, please cite the paper:

```bibtex
@article{DeNobili2026NamingGame,
  title         = {Microscopic dynamics of consensus formation in multi-agent {LLM} Naming Games},
  author        = {De Nobili, Cristiano and Iyer, Vijayasri and Codello, Alessandro and Burioni, Raffaella},
  year          = {2026},
  eprint        = {2608.02178},
  archivePrefix = {arXiv},
  primaryClass  = {physics.soc-ph}
}
```

Update the entry with the journal reference once the paper is published.

## License

Code: MIT. Data: CC BY 4.0.

## Contact

Cristiano De Nobili, cristiano@critiqality.ai
