# 12.5 Project opener visual remediation design

## 12.5.1 Purpose

Replace the repository's architecture-diagram opener with a deliberate project identity that is
clear and attractive on the repository README, the generated MkDocs landing page, and the GitHub
wiki. The result must preserve the three-surface documentation contract while making the project
name, purpose, stack, and Atlas-backed execution model legible before the first numbered section.

## 12.5.2 Evidence and design references

The current opener assigns `runtime-flow.png` two incompatible roles: it is both an architecture
diagram and the project poster. At README scale, its small topology labels become unreadable and its
tall technical composition overwhelms the opening prose. The existing checker then hard-codes that
asset as the required poster, so an aesthetically and semantically weak opener still passes CI.

Two references establish the replacement pattern:

- Atlas uses a dedicated full-width branded poster independently of its architecture diagrams and
  builds a richer, separately styled landing-page hero.
- Nativ uses a centered full-width banner, centered HTML H1 and descriptor lines, then a centered
  row of real shield images before its executive prose.

The implementation adopts the useful parts of both patterns without copying either project's
branding.

## 12.5.3 Considered approaches

### 12.5.3.1 Keep the runtime diagram and restyle only the surrounding Markdown

This is the smallest change, but it leaves the primary defect intact: a detailed systems diagram
is not a brand poster. Larger or centered typography cannot make its 8–11 pixel labels useful at
README scale. This approach is rejected.

### 12.5.3.2 Use a remote banner and remote shields

This closely matches common GitHub README patterns and requires little pipeline work. It also makes
all three surfaces depend on remote image hosts and weakens the repository's self-containment
contract. This approach is rejected for the final implementation.

### 12.5.3.3 Dedicated local poster plus local categorized shields

This is the selected approach. A new panoramic poster carries the visual identity and large
wordmark. Local SVG shield images provide the technology overview. The docs generator copies both
asset families into the site and wiki, while each hand-authored opener uses the path appropriate to
its source location. The runtime-flow diagram remains in the architecture page only.

## 12.5.4 Opener composition

The README and landing source use the same semantic sequence:

1. A centered, full-width local poster with meaningful alt text.
2. A centered HTML H1 carrying the project name.
3. A centered one-line tagline.
4. Centered badge rows grouped by `Core ML`, `NLP and graphs`, `Runtime`, and `Engineering`.
5. A 100–150-word executive summary split into two short paragraphs.

GitHub does not reliably preserve arbitrary inline font-size styling. The poster therefore carries
the intentionally oversized wordmark, while `<h1 align="center">` supplies an accessible,
selectable title on every surface. MkDocs adds narrowly scoped landing-page spacing, but the
opener must remain coherent without custom CSS so the repository and wiki are not second-class
surfaces.

## 12.5.5 Poster visual contract

The poster is a panoramic dark-first composition targeting a 2.4:1 display ratio. It uses
the repository's cyan, violet, emerald, and amber accents against a near-black technical-lab
background. A large `ML ENG LAB` wordmark is the dominant element. Supporting imagery evokes
notebook experimentation, neural networks, graphs, and an Atlas-backed remote runtime without
embedding small diagrams, terminal screenshots, detailed labels, logos from other projects, or
decorative marketing copy.

The asset must remain readable in both GitHub themes, avoid photographic people and vendor-logo
collages, include no watermark, and contain no text other than the exact project wordmark and the
short descriptor `NOTEBOOKS · SYSTEMS · REPRODUCIBILITY`.

## 12.5.6 Technology badge contract

Badge rows are grounded in committed configuration and active dependencies:

- **Core ML:** Python, Jupyter, NumPy, pandas, PyTorch, PyTorch Geometric, scikit-learn.
- **NLP and graphs:** spaCy, NLTK, NetworkX.
- **Runtime:** Atlas, Docker, VS Code, GitHub Codespaces.
- **Engineering:** NNx, Papermill, pytest, Ruff, GitHub Actions.

Badges are local SVG `<img>` assets, not a plain-text list. Their alt text names the technology,
and the surrounding prose labels each category so the visual remains understandable when images
are unavailable. The repository owns custom neutral shields for Atlas and NNx rather than implying
an official third-party logo.

## 12.5.7 Three-surface asset flow

Canonical poster and badge files live under `docs/assets/`. The site generator copies them to
`generated/site/assets/`, and the wiki generator copies them to `generated/wiki/assets/`. The root
README references `docs/assets/...`; `docs/index.md` references `assets/...`; generated paths remain
local to their destination surface. No surface links to another surface or relies on a repository
file-view URL.

The README and landing source remain hand-authored, so the checker normalizes their surface-specific
asset paths before comparing the shared header structure, tagline, badges, and summary.

## 12.5.8 Validation behavior

`check_project_opening` must require:

- the centered HTML title and poster wrapper;
- the dedicated poster path, with an explicit rejection of `runtime-flow`;
- all four badge categories and the complete expected badge set;
- real `<img>` tags whose local source files exist;
- the canonical tagline and identical normalized executive summary;
- a summary length of 100–150 words and exactly two prose paragraphs;
- opener order parity between README and landing source after path normalization.

Focused tests cover missing poster files, architecture-diagram substitution, left-aligned Markdown
titles, missing or plain-text badges, incomplete categories, summary drift, wrong paragraph count,
and invalid word counts. Existing determinism, self-containment, strict-build, and surface-copy
tests remain mandatory.

## 12.5.9 Documentation consequences

The runtime-flow diagram stays documented and embedded under the architecture page. The diagram
provenance page no longer describes it as an opener asset. The implementation record and changelog
state that the former poster was a reused architecture artifact and that the opener gate now checks
visual structure as well as string parity.

## 12.5.10 Acceptance criteria

1. README, generated site, and generated wiki open with the dedicated poster, centered title,
   centered tagline, grouped shield images, and two-paragraph executive summary.
2. No opener references `runtime-flow.png` or `runtime-flow.svg`.
3. The runtime-flow diagram remains present on the architecture page.
4. Every poster and badge reference resolves to a physical file within its own surface.
5. The opener checker rejects all defects listed in section 12.5.8.
6. `make docs-check`, the docs-script test suite, Ruff, the repository verifier, and deterministic
   site/wiki generation pass from a clean worktree.
7. The feature merges by PR into `develop`; a separate `develop` to `main` PR publishes Pages and
   the wiki.
