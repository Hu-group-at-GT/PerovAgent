from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from perovagent.chemistry.anchors import AnchorSpec
from perovagent.agents.structured_agent import StructuredAgent

log = logging.getLogger(__name__)


class Proposal(BaseModel):
    """A single candidate ligand with agent's rationale."""
    smiles: str
    anchor: str
    anchor_spec: Optional[AnchorSpec] = None
    hypothesis: str = ""
    predicted_direction: Literal["increase", "decrease"]
    parent_smiles: Optional[str] = None
    rationale: str = ""


class ProposalBatch(BaseModel):
    """The agent's per-round output as a list of Proposals."""
    proposals: list[Proposal] = Field(default_factory=list)


class LLMProposer(StructuredAgent):
    _output_type = ProposalBatch
    _preflight_probe = "Reply with an empty proposals list to confirm connectivity."

    def __init__(self, model: str, effort: str | None, system_context: str, *,
                 material: str, b_site: str, max_binding_energy: float,
                 effect_size_threshold: float):
        super().__init__(model, effort,
                         proposer_system_prompt(material, b_site, max_binding_energy,
                                                effect_size_threshold),
                         system_context)

    def propose(self, prompt: str) -> ProposalBatch:
        try:
            result = self.agent.run_sync(prompt)
        except Exception as e:
            log.warning("proposer agent call failed: %s", e)
            return ProposalBatch()
        self._accumulate(result)
        return result.output


def proposer_system_prompt(material: str, b_site: str, max_binding_energy: float,
                           effect_size_threshold: float) -> str:
    return f"""\
You design X-site passivant ligands for a {material} surface.

Objective and scope
- Find progressively stronger binders, where E_bind = E_full - E_vacancy - E_ligand and a more
  negative value is stronger. Values below about {max_binding_energy:g} eV are rejected as failed
  relaxations; near-zero or positive values remain informative measurements. Propose the ligand
  that tests your idea, not the one you expect to score well.
- Every energy here comes from a MatterSim+D3 machine-learned interatomic potential, a surrogate
  approximating DFT.
- Each E_bind is an average over several independent placements of the same molecule, so it
  carries pose-to-pose noise. Two molecules whose E_bind differ by less than {effect_size_threshold:g} eV
  are not distinguishable.
- Passivate only a halide vacancy or undercoordinated surface {b_site}. A-site passivation is
  excluded.
- Anionic mode uses a Brønsted-acid head deprotonated once to -1; neutral mode uses a Lewis-base
  donor without deprotonation. Explore both modes, and do not presume that any molecular property
  (anchor type, size, aromaticity, heteroatom placement) binds strongest; discover it from the
  measured data.
- Both modes are legal, and no anchor chemistry is named for you here on purpose: which heads
  exist, and which of them bind, is the design space you are here to map.

Anchor definitions
- Set the `anchor` field on every proposal: either a registered anchor name, or a new lowercase
  name that you also define with `anchor_spec`.
- Use a registered anchor name only when the molecule matches its exact registered motif.
- For a new lowercase anchor name, also supply `anchor_spec` with:
  - `binding_mode`: `"anionic"` or `"neutral"` -- how the head binds, not how you write it;
  - `motif_smarts`: the binding head as it appears on YOUR molecule written neutral and fully
    protonated;
  - `coord_atoms`: 1-2 donor-element symbols seated at the X-site -- two symbols for a
    bidentate head, one for a single donor;
  - for anionic mode, `deprot_smarts` (locates the acidic group) and `charge_target_index`
    (which atom IN THAT MATCH becomes -1 -- it indexes the `deprot_smarts` match, not the
    molecule; an in-range index read against the wrong base deprotonates the wrong atom and
    no validator catches it).
  The molecule must contain the motif and the specification must pass machine validation.
- Each molecule has exactly one binding head. A polyprotic head is one site and is allowed; the
  constructed bound form loses only one proton and other acidic protons remain protonated. Those
  groups remain chemically meaningful because they affect conformation, hydrogen bonding,
  contacts, and strain.
- `motif_smarts` must match the complete head exactly once. If a pattern matches once per acidic
  proton, include the full head in one pattern. Correct the SMARTS rather than changing the intended
  chemistry to satisfy an overly loose pattern.
- Registered motifs may include substituent atoms and be narrower than their names suggest. Check
  the listed motif after every edit. If it no longer matches, define a new anchor and specification.

Evidence
- Use the current prompt's round contract and context; calls do not share conversational state.
- Follow the size and allowed-element limits in the admission constraints block below; a molecule
  outside them is rejected before evaluation.
- When available, context includes strong, mid-pack, and weak surrogate estimates, per-anchor and
  scaffold summaries, a learned edit playbook, and relaxed-slab contacts formatted as
  `L<id>{{atom: <element>@<distance A>, ...}}`, where `*` marks the surface anchor.
- Treat measured energies and geometry as evidence. Treat proposed causal explanations as
  hypotheses unless independently tested.
- 2D structure barely predicts binding energy. Reason from the measured energies and relaxed
  geometry rather than from structural priors. No worked example is given anywhere in this
  prompt, on purpose: any example would itself be a prior about the answer, and which donors and
  mechanisms actually work here is what you are measuring.
- Each round, use the measured evidence block to check whether your last proposals' directional
  hypotheses held, and carry that outcome into this round's batch.
- Principle status matters: `candidate` is retrospective and in-sample; `supported` is
  prospectively confirmed by the surrogate workflow.

Round handling
- The current prompt begins with the delivery count, selected explore/exploit/explain mode, and
  round-specific brief. Follow that contract. Exact repeats are rejected; new close analogs remain
  eligible.
- An explore round asks for a diverse batch that both tests your current hypotheses and reaches
  sparse regions, spanning many anchor chemistries -- anionic and neutral, including new anchors
  you define -- and many scaffold types: monocyclic, fused, bridged, spiro, cage, saturated, and
  flexible/acyclic. Do not assume any class binds strongest.
- An exploit round refines the named strongest binders. When the brief asks for matched single
  edits, each child differs from one named parent by exactly one structural edit with everything
  else preserved, and its `parent_smiles` is that exact parent, so the change in binding energy is
  attributable to that edit. Otherwise propose free close analogs of the best hits.
- An explain round asks for matched satisfy/violate pairs. Propose each satisfying molecule
  together with a near-identical violating twin that differs only in the predicate feature,
  holding anchor, scaffold, and size fixed wherever the predicate permits. If the predicate is
  itself the anchor head, the twin must use a different head; then match scaffold and size as
  closely as you can. Set each violating twin's `parent_smiles` to the satisfying molecule it
  pairs with: that link is what makes the round a paired test. Unlinked arms are scored as two
  independently designed sets, which throws away the matching you just built and leaves the
  scaffold and size differences between them uncontrolled.
- Stereochemistry is removed before a molecule is identified, so stereoisomers are the same
  molecule here: a stereo variant is not a distinct candidate, not a single edit, and not a valid
  twin. Write `parent_smiles` without stereo markers.

Predictions and output
- Do not predict a numerical binding energy.
- For each proposal, provide `hypothesis`, `predicted_direction`, and, when applicable,
  `parent_smiles` and `anchor_spec`.
- `hypothesis` names the changed feature, the contact it should make with the surface, the
  expected direction, and the proposed mechanism, while keeping observation separate from
  interpretation. A hypothesis stated this way is falsifiable by the next measurement.
- `predicted_direction` is `"increase"` or `"decrease"` in binding strength; `"increase"` means
  stronger binding and a more negative E_bind.
- `parent_smiles` is the reference this molecule is measured against; prefer one shown in the
  evidence block. An explain round is the exception: a violating twin's parent is the satisfying
  molecule it pairs with in the same batch, which has not been measured yet. Leave it null only
  for a de-novo choice. With a parent, the directional call is
  scored against that measured reference. Without one, `predicted_direction` is only an
  unscored qualitative expectation relative to the campaign's current typical binders, or a
  prior expectation on a cold start; state that uncertainty in `hypothesis` rather than
  implying a measured comparison.
- If a prior proposal was rejected, return a different valid molecule."""
