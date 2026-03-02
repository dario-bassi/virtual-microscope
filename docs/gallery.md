# Virtual Microscope — Backend Gallery

Showcase images and short description for **35** backends.

Dynamics: ![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![static](https://img.shields.io/badge/static-CBCBCC)
Available imaging channels: ![channel](https://img.shields.io/badge/channel-ADD3FF)
Extra hardware devices: ![device](https://img.shields.io/badge/device-DBC8FF)

---

## bacteria

*E. coli (rod-shaped bacteria) — Phase-contrast + epifluorescence*

Simulates rod-shaped bacteria with growth and division dynamics. Supports SLM-based optogenetic stimulation and temperature control.

<p style="display:flex;gap:4px"><img src="gallery/frames/bacteria_01.png" width="24%"> <img src="gallery/frames/bacteria_02.png" width="24%"> <img src="gallery/frames/bacteria_03.png" width="24%"> <img src="gallery/frames/bacteria_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![GFP](https://img.shields.io/badge/GFP-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![SLM](https://img.shields.io/badge/SLM-DBC8FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF)

**Experiment guide:** Seed a population of bacteria and watch them grow and divide. Use phase-contrast for morphology and GFP/DAPI for fluorescence markers. Apply SLM masks to trigger optogenetic stimulation in specific regions. Raise temperature to accelerate growth or cool to 4°C to arrest division.

**Devices:** SLM — Activates bPAC optogenetic tool in illuminated cells, increasing cAMP | Temperature — Growth rate scales with temperature; 4°C arrests division, 42°C is heat shock

**Parameters:** `n_cells` (Initial cell count (default 30)) · `world_size` (World size in pixels (default 512))

## blood_smear

*Peripheral blood smear (Wright-Giemsa stain) — Brightfield histology (RGB output)*

Simulates a stained peripheral blood smear with RBCs, WBCs, and platelets. Supports abnormal morphologies and rouleaux formation.

<p style="display:flex;gap:4px"><img src="gallery/frames/blood_smear_01.png" width="24%"> <img src="gallery/frames/blood_smear_02.png" width="24%"> <img src="gallery/frames/blood_smear_03.png" width="24%"> <img src="gallery/frames/blood_smear_04.png" width="24%"></p>

![static](https://img.shields.io/badge/static-CBCBCC) ![wright-giemsa](https://img.shields.io/badge/wright--giemsa-ADD3FF) ![nuclei-aid](https://img.shields.io/badge/nuclei--aid-ADD3FF) ![membrane-aid](https://img.shields.io/badge/membrane--aid-ADD3FF)

**Experiment guide:** A fixed, stained peripheral blood smear. Navigate the slide to find RBCs, WBCs (neutrophils, lymphocytes, monocytes), and platelets. Switch between composite Wright-Giemsa view and auxiliary channels highlighting nuclei or membranes. Useful for training automated differential white-cell counters and detecting abnormal morphologies.

**Parameters:** `n_rbc` (Red blood cell count (default 800)) · `n_wbc` (White blood cell count (default 15)) · `abnormal_rbc` (Dict of abnormality fractions (e.g. sickle, target)) · `rouleaux_fraction` (Fraction of RBCs in rouleaux stacks (default 0.0))

## calcium

*Epithelial monolayer with GCaMP calcium reporter — Phase-contrast + epifluorescence*

Simulates calcium wave propagation across an epithelial cell monolayer. Features GCaMP reporter with SLM-triggered stimulation and temperature control.

<p style="display:flex;gap:4px"><img src="gallery/frames/calcium_01.png" width="24%"> <img src="gallery/frames/calcium_02.png" width="24%"> <img src="gallery/frames/calcium_03.png" width="24%"> <img src="gallery/frames/calcium_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![GCaMP](https://img.shields.io/badge/GCaMP-ADD3FF) ![E-cadherin](https://img.shields.io/badge/E--cadherin-ADD3FF) ![SLM](https://img.shields.io/badge/SLM-DBC8FF) ![SLM-Mode](https://img.shields.io/badge/SLM--Mode-DBC8FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF)

**Experiment guide:** Observe calcium wave propagation across a cell monolayer using the GCaMP fluorescent reporter. Use SLM masks to trigger calcium release at specific locations and watch the wave spread through gap junctions. Switch SLM-Mode between stimulation and spatial-filter illumination. Temperature modulates wave speed and excitability.

**Devices:** SLM — Triggers calcium release in illuminated cells, initiating a propagating wave | SLM-Mode — Toggles between optogenetic stimulation and spatial-filter illumination | Temperature — Higher temperature increases wave speed and cell excitability

**Parameters:** `n_cells` (Number of cells in the monolayer (default 200)) · `grid_size` (Simulation grid size in pixels (default 512)) · `Du` (Calcium diffusion coefficient (default 5.0))

## cardio

*iPSC-derived cardiomyocytes — Phase-contrast + epifluorescence*

Simulates beating cardiomyocytes with calcium transients and arrhythmia. Supports SLM pacing, temperature control, and perfusion.

<p style="display:flex;gap:4px"><img src="gallery/frames/cardio_01.png" width="24%"> <img src="gallery/frames/cardio_02.png" width="24%"> <img src="gallery/frames/cardio_03.png" width="24%"> <img src="gallery/frames/cardio_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![GCaMP](https://img.shields.io/badge/GCaMP-ADD3FF) ![cell-junctions](https://img.shields.io/badge/cell--junctions-ADD3FF) ![SLM](https://img.shields.io/badge/SLM-DBC8FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Perfusion](https://img.shields.io/badge/Perfusion-DBC8FF)

**Experiment guide:** Observe beating cardiomyocytes with calcium transients visualised via GCaMP. Normal cells beat synchronously; a fraction exhibit arrhythmic pacing. Use SLM to optically pace specific regions and restore synchrony. Perfusion delivers drugs (e.g. isoproterenol) that alter beat rate. Temperature affects contraction frequency.

**Devices:** SLM — Optically paces illuminated cells, overriding their intrinsic rhythm | Temperature — Beat frequency scales with temperature (Q10 ≈ 2) | Perfusion — Delivers chronotropic drugs that increase or decrease beat rate

**Parameters:** `n_cells` (Number of cardiomyocytes (default 300)) · `normal_freq` (Normal beating frequency in Hz (default 1.0)) · `arrhythmia_fraction` (Fraction of cells with arrhythmic pacing (default 0.12))

## celegans

*C. elegans (nematode worm) — DIC + epifluorescence*

Simulates a crawling C. elegans nematode with sinusoidal body bending. Features pharyngeal GFP and body-wall mCherry with temperature, perfusion, and electrode control.

<p style="display:flex;gap:4px"><img src="gallery/frames/celegans_01.png" width="24%"> <img src="gallery/frames/celegans_02.png" width="24%"> <img src="gallery/frames/celegans_03.png" width="24%"> <img src="gallery/frames/celegans_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![DIC](https://img.shields.io/badge/DIC-ADD3FF) ![GFP-pharynx](https://img.shields.io/badge/GFP--pharynx-ADD3FF) ![mCherry-body](https://img.shields.io/badge/mCherry--body-ADD3FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Perfusion](https://img.shields.io/badge/Perfusion-DBC8FF)

**Experiment guide:** Track a crawling C. elegans nematode with sinusoidal body bending. Use DIC for body morphology, GFP-pharynx for the feeding organ, and mCherry-body for body-wall muscle. Temperature affects crawling speed; perfusion can deliver paralytic agents (e.g. levamisole).

**Devices:** Temperature — Crawling speed scales with temperature; low temperature slows locomotion | Perfusion — Delivers paralytic agents or nutrients affecting worm behaviour

**Parameters:** `worm_length` (Worm body length in pixels (default 250)) · `worm_width` (Worm body width in pixels (default 18)) · `speed` (Crawling speed in px/s (default 40))

## colony_counter

*Bacterial colony plate (agar) — Brightfield plate imaging*

Simulates bacterial colony plates for colony counting assays. Supports spread/streak plate types with optional blue-white screening and GFP.

<p style="display:flex;gap:4px"><img src="gallery/frames/colony_counter_01.png" width="24%"> <img src="gallery/frames/colony_counter_02.png" width="24%"> <img src="gallery/frames/colony_counter_03.png" width="24%"> <img src="gallery/frames/colony_counter_04.png" width="24%"></p>

![static](https://img.shields.io/badge/static-CBCBCC) ![plate-image](https://img.shields.io/badge/plate--image-ADD3FF) ![blue-channel](https://img.shields.io/badge/blue--channel-ADD3FF) ![gfp-channel](https://img.shields.io/badge/gfp--channel-ADD3FF)

**Experiment guide:** A bacterial colony plate viewed from above. Count colonies on spread or streak plates, with optional blue-white screening (lacZ) or GFP fluorescence. Useful for training colony-counting algorithms and studying plating efficiency, satellite colonies, and zone-of-inhibition assays.

**Parameters:** `n_colonies` (Number of colonies (default 200)) · `plate_type` (Plate type: 'spread' or 'streak' (default 'spread')) · `staining` (Optional staining: 'blue-white', 'gfp', or None)

## dictyostelium

*Dictyostelium discoideum (social amoeba) — Dark-field + epifluorescence*

Simulates Dictyostelium discoideum chemotaxis with cAMP wave relay. Features pacemaker-driven aggregation with SLM stimulation and temperature control.

<p style="display:flex;gap:4px"><img src="gallery/frames/dictyostelium_01.png" width="24%"> <img src="gallery/frames/dictyostelium_02.png" width="24%"> <img src="gallery/frames/dictyostelium_03.png" width="24%"> <img src="gallery/frames/dictyostelium_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![dark-field](https://img.shields.io/badge/dark--field-ADD3FF) ![GFP](https://img.shields.io/badge/GFP-ADD3FF) ![cAMP-reporter](https://img.shields.io/badge/cAMP--reporter-ADD3FF) ![SLM](https://img.shields.io/badge/SLM-DBC8FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF)

**Experiment guide:** Watch Dictyostelium amoebae aggregate via cAMP chemotaxis. Pacemaker cells emit periodic cAMP pulses that are relayed outward as spiral waves, guiding cells toward aggregation centres. Use SLM to create artificial cAMP sources and redirect streaming. Temperature modulates relay kinetics and aggregation speed.

**Devices:** SLM — Creates artificial cAMP point sources in illuminated regions | Temperature — Modulates cAMP relay kinetics and cell motility speed

**Parameters:** `n_cells` (Number of amoebae (default 100)) · `n_pacemakers` (Number of pacemaker cells (default 3)) · `relay_radius` (cAMP relay radius in pixels (default 40))

## fibroblast

*Adherent fibroblasts (primary or cell line) — Brightfield + epifluorescence*

Simulates adherent fibroblasts with actin stress fibres and focal adhesions. Supports temperature, perfusion, and mechanical stretch.

<p style="display:flex;gap:4px"><img src="gallery/frames/fibroblast_01.png" width="24%"> <img src="gallery/frames/fibroblast_02.png" width="24%"> <img src="gallery/frames/fibroblast_03.png" width="24%"> <img src="gallery/frames/fibroblast_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![brightfield](https://img.shields.io/badge/brightfield-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![phalloidin](https://img.shields.io/badge/phalloidin-ADD3FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Perfusion](https://img.shields.io/badge/Perfusion-DBC8FF) ![Stretch](https://img.shields.io/badge/Stretch-DBC8FF)

**Experiment guide:** Observe adherent fibroblasts with actin stress fibres (phalloidin) and nuclear staining (DAPI). Apply mechanical stretch to study cytoskeletal remodelling. Perfusion delivers cytochalasin-D or latrunculin-A to disrupt the actin network. Temperature affects cell spreading and migration.

**Devices:** Temperature — Higher temperature increases cell spreading and migration speed | Perfusion — Delivers actin-disrupting drugs (CytoD, LatA) that dissolve stress fibres | Stretch — Applies uniaxial mechanical stretch, reorienting stress fibres perpendicular to strain

**Parameters:** `n_cells` (Number of fibroblasts (default 8)) · `world_size` (World size in pixels (default 512))

## fish

*Tissue section with FISH probes — Phase-contrast + epifluorescence*

Simulates fluorescence in situ hybridisation (FISH) with punctate probe signals in nuclei. Supports gene amplification and deletion with temperature, perfusion, and electrode.

<p style="display:flex;gap:4px"><img src="gallery/frames/fish_01.png" width="24%"> <img src="gallery/frames/fish_02.png" width="24%"> <img src="gallery/frames/fish_03.png" width="24%"> <img src="gallery/frames/fish_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![membrane](https://img.shields.io/badge/membrane-ADD3FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Perfusion](https://img.shields.io/badge/Perfusion-DBC8FF) ![Electrode](https://img.shields.io/badge/Electrode-DBC8FF)

**Experiment guide:** Examine fluorescence in situ hybridisation (FISH) signals in an epithelial tissue section. Each nucleus contains punctate probe signals indicating gene copy number. A fraction of cells show amplification (extra copies) or deletion (fewer copies). Useful for training HER2-FISH scoring algorithms.

**Devices:** Temperature — No biological effect (static specimen); adjusts thermal noise | Perfusion — No biological effect on fixed tissue | Electrode — No biological effect on fixed tissue

**Parameters:** `n_cells` (Number of cells in the tissue section (default 40)) · `locus_copies` (Normal gene copy number (default 2)) · `amplified_fraction` (Fraction of cells with gene amplification (default 0.15))

## flow_cytometry

*Cell suspension (multi-population) — Flow cytometry scatter + fluorescence*

Simulates a flow cytometer producing scatter and fluorescence dot-plot images. Generates multi-population event data with FITC and PE channels.

<p style="display:flex;gap:4px"><img src="gallery/frames/flow_cytometry_01.png" width="24%"> <img src="gallery/frames/flow_cytometry_02.png" width="24%"> <img src="gallery/frames/flow_cytometry_03.png" width="24%"> <img src="gallery/frames/flow_cytometry_04.png" width="24%"></p>

![static](https://img.shields.io/badge/static-CBCBCC) ![scatter](https://img.shields.io/badge/scatter-ADD3FF) ![FITC](https://img.shields.io/badge/FITC-ADD3FF) ![PE](https://img.shields.io/badge/PE-ADD3FF)

**Experiment guide:** Simulates a flow cytometer acquiring scatter and fluorescence events from a mixed cell population. Each snap adds a batch of events to the dot-plot image. Observe distinct populations in FSC/SSC scatter and FITC/PE fluorescence channels. Useful for training gating and clustering algorithms.

**Parameters:** `n_total` (Total events to acquire (default 10000)) · `events_per_snap` (Events added per snap (default 100))

## fucci

*Epithelial cells with FUCCI cell-cycle reporter — Phase-contrast + epifluorescence*

Simulates FUCCI cell-cycle reporter with mCherry-Cdt1 (G1) and mVenus-Geminin (S/G2/M). Features configurable phase durations with temperature and perfusion.

<p style="display:flex;gap:4px"><img src="gallery/frames/fucci_01.png" width="24%"> <img src="gallery/frames/fucci_02.png" width="24%"> <img src="gallery/frames/fucci_03.png" width="24%"> <img src="gallery/frames/fucci_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![mCherry-Cdt1](https://img.shields.io/badge/mCherry--Cdt1-ADD3FF) ![mVenus-Geminin](https://img.shields.io/badge/mVenus--Geminin-ADD3FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Perfusion](https://img.shields.io/badge/Perfusion-DBC8FF)

**Experiment guide:** Track cell-cycle progression using the FUCCI dual-colour reporter. G1-phase cells express mCherry-Cdt1 (red) while S/G2/M cells express mVenus-Geminin (green). Watch the monolayer as cells cycle and divide. Temperature accelerates or slows cell-cycle progression; perfusion can deliver cell-cycle inhibitors.

**Devices:** Temperature — Higher temperature accelerates cell-cycle progression | Perfusion — Delivers cell-cycle inhibitors (e.g. thymidine block, nocodazole)

**Parameters:** `n_cells` (Initial cell count (default 60)) · `g1_duration` (G1 phase duration in steps (default 30)) · `s_duration` (S phase duration in steps (default 15))

## gel_doc

*Electrophoresis gel (western, DNA agarose, or Coomassie) — Gel documentation (transillumination / chemiluminescence)*

Simulates a gel electrophoresis documentation system for western or DNA gels. Renders multi-lane band patterns with configurable lane count and gel type.

<p style="display:flex;gap:4px"><img src="gallery/frames/gel_doc_01.png" width="24%"> <img src="gallery/frames/gel_doc_02.png" width="24%"> <img src="gallery/frames/gel_doc_03.png" width="24%"> <img src="gallery/frames/gel_doc_04.png" width="24%"></p>

![static](https://img.shields.io/badge/static-CBCBCC) ![gel-image](https://img.shields.io/badge/gel--image-ADD3FF)

**Experiment guide:** A gel electrophoresis documentation image. View multi-lane band patterns from western blots, DNA agarose gels, or Coomassie-stained protein gels. Navigate the gel with XY stage and zoom with objectives. Useful for training lane-detection and band-quantification pipelines.

**Parameters:** `n_lanes` (Number of gel lanes (default 8)) · `gel_type` (Gel type: 'western', 'dna', or 'coomassie' (default 'western'))

## hemocytometer

*Cell suspension in Neubauer counting chamber — Brightfield with trypan blue*

Simulates a hemocytometer counting chamber with trypan-blue viability staining. Supports cell clumps and configurable dilution factor.

<p style="display:flex;gap:4px"><img src="gallery/frames/hemocytometer_01.png" width="24%"> <img src="gallery/frames/hemocytometer_02.png" width="24%"> <img src="gallery/frames/hemocytometer_03.png" width="24%"> <img src="gallery/frames/hemocytometer_04.png" width="24%"></p>

![static](https://img.shields.io/badge/static-CBCBCC) ![brightfield](https://img.shields.io/badge/brightfield-ADD3FF) ![trypan-blue](https://img.shields.io/badge/trypan--blue-ADD3FF)

**Experiment guide:** A hemocytometer counting chamber loaded with a cell suspension and trypan-blue viability dye. Live cells exclude the dye and appear bright; dead cells stain blue. Count cells in the grid squares to estimate concentration. Useful for training cell-counting and viability-estimation algorithms.

**Parameters:** `n_cells` (Total cells in the chamber (default 150)) · `viability` (Fraction of live cells (default 0.85)) · `dilution_factor` (Dilution factor for concentration calculation (default 2))

## histology

*H&E-stained tissue section — Brightfield histopathology (RGB output)*

Simulates H&E-stained histology tissue sections with configurable tissue type and tumour grade. Supports separate hematoxylin and eosin pseudo-channels.

<p style="display:flex;gap:4px"><img src="gallery/frames/histology_01.png" width="24%"> <img src="gallery/frames/histology_02.png" width="24%"> <img src="gallery/frames/histology_03.png" width="24%"> <img src="gallery/frames/histology_04.png" width="24%"></p>

![static](https://img.shields.io/badge/static-CBCBCC) ![H-and-E](https://img.shields.io/badge/H--and--E-ADD3FF) ![hematoxylin](https://img.shields.io/badge/hematoxylin-ADD3FF) ![eosin](https://img.shields.io/badge/eosin-ADD3FF)

**Experiment guide:** A static stained tissue section. Switch between composite H&E view, isolated hematoxylin (nuclei), and eosin (cytoplasm/stroma) channels. Navigate across the slide with XY stage and zoom with objectives. Useful for training histopathology image analysis pipelines.

**Parameters:** `tissue_type` (Tissue morphology preset (default 'glandular')) · `grade` (Tumour differentiation grade (default 0 = normal)) · `n_nuclei` (Number of nuclei to generate (default 200))

## lipid_droplet

*Hepatocytes with intracellular lipid droplets — Phase-contrast + epifluorescence*

Simulates intracellular lipid droplets stained with BODIPY for hepatic steatosis studies. Features steatotic and normal cell populations with temperature and perfusion.

<p style="display:flex;gap:4px"><img src="gallery/frames/lipid_droplet_01.png" width="24%"> <img src="gallery/frames/lipid_droplet_02.png" width="24%"> <img src="gallery/frames/lipid_droplet_03.png" width="24%"> <img src="gallery/frames/lipid_droplet_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![membrane](https://img.shields.io/badge/membrane-ADD3FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Perfusion](https://img.shields.io/badge/Perfusion-DBC8FF)

**Experiment guide:** Observe intracellular lipid droplets stained with BODIPY in a hepatocyte monolayer. A configurable fraction of cells are steatotic (fatty) with abundant large droplets. Track lipid accumulation over time. Temperature affects metabolic rate; perfusion delivers lipogenic or lipolytic drugs.

**Devices:** Temperature — Higher temperature increases lipid metabolism rate | Perfusion — Delivers oleic acid (lipogenic) or forskolin (lipolytic) to modulate steatosis

**Parameters:** `n_cells` (Number of hepatocytes (default 25)) · `steatotic_fraction` (Fraction of steatotic cells (default 0.4))

## lysosome

*Epithelial cells with LysoTracker-stained lysosomes — Phase-contrast + epifluorescence*

Simulates lysosomal trafficking with LysoTracker-stained puncta undergoing Brownian motion. Features configurable lysosome count and diffusion rate with temperature and perfusion.

<p style="display:flex;gap:4px"><img src="gallery/frames/lysosome_01.png" width="24%"> <img src="gallery/frames/lysosome_02.png" width="24%"> <img src="gallery/frames/lysosome_03.png" width="24%"> <img src="gallery/frames/lysosome_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![membrane](https://img.shields.io/badge/membrane-ADD3FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Perfusion](https://img.shields.io/badge/Perfusion-DBC8FF)

**Experiment guide:** Watch LysoTracker-stained lysosomes undergoing Brownian diffusion inside epithelial cells. Track individual puncta over time for single-particle analysis. Temperature modulates diffusion rate; perfusion delivers agents that alter lysosomal trafficking (e.g. chloroquine, bafilomycin).

**Devices:** Temperature — Higher temperature increases lysosomal diffusion rate | Perfusion — Delivers agents that alter lysosomal pH or trafficking (chloroquine, bafilomycin)

**Parameters:** `n_cells` (Number of cells (default 20)) · `n_lyso_min` (Minimum lysosomes per cell (default 5)) · `n_lyso_max` (Maximum lysosomes per cell (default 20)) · `diffusion_rate` (Brownian diffusion coefficient (default 0.3))

## malaria

*Giemsa-stained thin blood smear with Plasmodium parasites — Brightfield histology (RGB output)*

Simulates a Giemsa-stained malaria blood smear with intra-erythrocytic Plasmodium parasites. Features configurable parasitemia, stage progression, and appliqué forms.

<p style="display:flex;gap:4px"><img src="gallery/frames/malaria_01.png" width="24%"> <img src="gallery/frames/malaria_02.png" width="24%"> <img src="gallery/frames/malaria_03.png" width="24%"> <img src="gallery/frames/malaria_04.png" width="24%"></p>

![static](https://img.shields.io/badge/static-CBCBCC) ![giemsa](https://img.shields.io/badge/giemsa-ADD3FF) ![chromatin-aid](https://img.shields.io/badge/chromatin--aid-ADD3FF) ![RBC-overlay](https://img.shields.io/badge/RBC--overlay-ADD3FF)

**Experiment guide:** A Giemsa-stained thin blood smear with intra-erythrocytic Plasmodium parasites at various developmental stages (ring, trophozoite, schizont). Navigate the smear to find infected RBCs and count parasitemia. Useful for training malaria diagnostic algorithms and studying parasite morphology including appliqué (accolé) forms.

**Parameters:** `parasitemia` (Fraction of infected RBCs (default 0.05)) · `n_rbc` (Total red blood cell count (default 2000)) · `hours_per_step` (Hours of parasite development per step (default 2.0))

## microfluidics

*Cells in a microfluidic channel — Phase-contrast + epifluorescence*

Simulates cells flowing through a microfluidic channel with optional traps and chemical gradients. Supports temperature and perfusion control.

<p style="display:flex;gap:4px"><img src="gallery/frames/microfluidics_01.png" width="24%"> <img src="gallery/frames/microfluidics_02.png" width="24%"> <img src="gallery/frames/microfluidics_03.png" width="24%"> <img src="gallery/frames/microfluidics_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![fluorescein](https://img.shields.io/badge/fluorescein-ADD3FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Perfusion](https://img.shields.io/badge/Perfusion-DBC8FF)

**Experiment guide:** Watch cells flowing through a microfluidic channel with optional trapping posts and chemical gradients. Use perfusion to control flow speed and deliver fluorescein for gradient visualisation. Temperature affects cell viability and motility. Useful for studying shear stress effects and chemotaxis in confined geometries.

**Devices:** Temperature — Affects cell viability and motility in the channel | Perfusion — Controls flow speed and delivers chemical gradients

**Parameters:** `n_cells` (Number of cells in the channel (default 30)) · `channel_width` (Channel width in pixels (default 100)) · `flow_speed` (Flow speed in px/step (default 3.0)) · `n_traps` (Number of trapping posts (default 0))

## mito

*Cultured cells with MitoTracker-stained mitochondria — Brightfield + epifluorescence*

Simulates mitochondrial networks with tubules undergoing fission and fusion. Features MitoTracker staining with configurable fragmentation dynamics.

<p style="display:flex;gap:4px"><img src="gallery/frames/mito_01.png" width="24%"> <img src="gallery/frames/mito_02.png" width="24%"> <img src="gallery/frames/mito_03.png" width="24%"> <img src="gallery/frames/mito_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![brightfield](https://img.shields.io/badge/brightfield-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![MitoTracker](https://img.shields.io/badge/MitoTracker-ADD3FF)

**Experiment guide:** Observe mitochondrial networks stained with MitoTracker. Tubules undergo stochastic fission and fusion, dynamically remodelling the network. Increase fragmentation to simulate stress or drug treatment (e.g. CCCP). Track network topology changes over time for morphometric analysis.

**Parameters:** `n_tubules` (Initial number of mitochondrial tubules (default 40)) · `fragmentation` (Baseline fragmentation level 0–1 (default 0.0)) · `fission_rate` (Fission event probability per step (default 0.03)) · `fusion_rate` (Fusion event probability per step (default 0.03))

## neuron

*Primary cortical neurons in culture — Phase-contrast + epifluorescence*

Simulates cultured neurons with branching dendrites and axons. Features MAP2-GFP and synaptophysin markers for neuronal morphology.

<p style="display:flex;gap:4px"><img src="gallery/frames/neuron_01.png" width="24%"> <img src="gallery/frames/neuron_02.png" width="24%"> <img src="gallery/frames/neuron_03.png" width="24%"> <img src="gallery/frames/neuron_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![MAP2-GFP](https://img.shields.io/badge/MAP2--GFP-ADD3FF) ![synaptophysin](https://img.shields.io/badge/synaptophysin-ADD3FF)

**Experiment guide:** Observe cultured neurons with branching dendrites and axons. MAP2-GFP labels dendrites; synaptophysin marks presynaptic terminals. Track neurite outgrowth and synapse formation over time. Useful for training neurite-tracing and synapse-detection algorithms.

**Parameters:** `n_neurons` (Number of neurons (default 8)) · `world_size` (World size in pixels (default 512))

## optogenetic

*Vertex-model epithelial cells with optogenetic actuator — Phase-contrast + epifluorescence*

Simulates vertex-based cells with optogenetic SLM stimulation driving cell motility. Uses ScatteredCellSim with light-activated morphological responses.

<p style="display:flex;gap:4px"><img src="gallery/frames/optogenetic_01.png" width="24%"> <img src="gallery/frames/optogenetic_02.png" width="24%"> <img src="gallery/frames/optogenetic_03.png" width="24%"> <img src="gallery/frames/optogenetic_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![membrane](https://img.shields.io/badge/membrane-ADD3FF) ![SLM](https://img.shields.io/badge/SLM-DBC8FF)

**Experiment guide:** Vertex-based cells respond to SLM illumination with increased motility and morphological changes. Draw SLM masks to selectively activate cells and observe directed migration. Useful for studying optogenetic control of cell mechanics and collective migration.

**Devices:** SLM — Activates optogenetic actuator in illuminated cells, increasing contractility and motility

**Parameters:** `n_cells` (Number of cells (default 30)) · `world_size` (World size in pixels (default 600)) · `base_radius` (Base cell radius in pixels (default 20))

## organoid

*Intestinal organoid cross-section — Brightfield + epifluorescence*

Simulates a 3D organoid cross-section with lumen, wall cells, and optional budding. Features E-cadherin junctional staining and DAPI nuclear label.

<p style="display:flex;gap:4px"><img src="gallery/frames/organoid_01.png" width="24%"> <img src="gallery/frames/organoid_02.png" width="24%"> <img src="gallery/frames/organoid_03.png" width="24%"> <img src="gallery/frames/organoid_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![brightfield](https://img.shields.io/badge/brightfield-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![E-cadherin](https://img.shields.io/badge/E--cadherin-ADD3FF)

**Experiment guide:** View a cross-section through a 3D organoid with a central lumen, polarised wall cells, and optional budding crypts. E-cadherin highlights cell–cell junctions; DAPI labels nuclei. Track organoid growth and morphogenesis over time. Useful for training organoid segmentation and morphometric analysis pipelines.

**Parameters:** `outer_radius` (Organoid outer radius in pixels (default 120)) · `n_cells` (Number of cells in the organoid wall (default 400)) · `n_buds` (Number of crypt buds (default 0))

## particle

*Generic scattered cells (vertex model) — Phase-contrast + epifluorescence*

Simulates scattered vertex-based cells as generic particles for basic microscopy. Provides phase-contrast, DAPI, and membrane channels.

<p style="display:flex;gap:4px"><img src="gallery/frames/particle_01.png" width="24%"> <img src="gallery/frames/particle_02.png" width="24%"> <img src="gallery/frames/particle_03.png" width="24%"> <img src="gallery/frames/particle_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![membrane](https://img.shields.io/badge/membrane-ADD3FF)

**Experiment guide:** A generic cell simulation using the vertex-based scattered-cell model. Cells undergo cell-cycle progression with division and apoptosis. Provides phase-contrast, DAPI, and membrane channels as a baseline for testing image-analysis pipelines on simple cell populations.

**Parameters:** `n_cells` (Initial cell count (default 50)) · `world_size` (World size in pixels (default 1500)) · `base_radius` (Base cell radius in pixels (default 20))

## plant_cell

*Plant tissue (e.g. onion epidermis, Elodea leaf) — Brightfield (iodine) + epifluorescence*

Simulates rectangular plant cells with cell walls, chloroplasts, and vacuoles. Features iodine staining with DAPI and Calcofluor-White channels.

<p style="display:flex;gap:4px"><img src="gallery/frames/plant_cell_01.png" width="24%"> <img src="gallery/frames/plant_cell_02.png" width="24%"> <img src="gallery/frames/plant_cell_03.png" width="24%"> <img src="gallery/frames/plant_cell_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![iodine-stain](https://img.shields.io/badge/iodine--stain-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![Calcofluor-White](https://img.shields.io/badge/Calcofluor--White-ADD3FF)

**Experiment guide:** View rectangular plant cells with rigid cell walls, chloroplasts, and central vacuoles. Iodine staining highlights starch granules; Calcofluor-White labels cell walls; DAPI labels nuclei. Useful for teaching plant cell anatomy and training cell-wall segmentation algorithms.

**Parameters:** `world_size` (World size in pixels (default 512)) · `cell_length_range` (Cell length range in px (default (150, 250))) · `wall_thickness` (Cell wall thickness in px (default 7.0))

## plate_reader

*96-well microplate (absorbance / fluorescence) — Plate reader heatmap*

Simulates a microplate reader producing well-based absorbance or fluorescence heatmaps. Channels vary by assay preset (viability, ELISA, Bradford, etc.).

<p style="display:flex;gap:4px"><img src="gallery/frames/plate_reader_01.png" width="24%"> <img src="gallery/frames/plate_reader_02.png" width="24%"> <img src="gallery/frames/plate_reader_03.png" width="24%"> <img src="gallery/frames/plate_reader_04.png" width="24%"></p>

![static](https://img.shields.io/badge/static-CBCBCC)

**Experiment guide:** Simulates a microplate reader producing well-based absorbance or fluorescence readings displayed as a heatmap. Choose from assay presets (viability, ELISA, Bradford, kinase, etc.) that set appropriate wavelengths and dose–response curves. Useful for training plate-layout analysis and hit-calling pipelines.

**Parameters:** `assay_type` (Assay preset: 'viability', 'elisa', 'bradford', etc. (default 'viability'))

## reaction_diffusion

*Gray–Scott reaction–diffusion system — Pseudo-fluorescence (activator / inhibitor concentration maps)*

Simulates Gray-Scott reaction-diffusion Turing patterns with activator and inhibitor species. Supports SLM-based perturbation with configurable F/K parameters.

<p style="display:flex;gap:4px"><img src="gallery/frames/reaction_diffusion_01.png" width="24%"> <img src="gallery/frames/reaction_diffusion_02.png" width="24%"> <img src="gallery/frames/reaction_diffusion_03.png" width="24%"> <img src="gallery/frames/reaction_diffusion_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![both-species](https://img.shields.io/badge/both--species-ADD3FF) ![activator-U](https://img.shields.io/badge/activator--U-ADD3FF) ![inhibitor-V](https://img.shields.io/badge/inhibitor--V-ADD3FF) ![SLM](https://img.shields.io/badge/SLM-DBC8FF) ![SLM-Mode](https://img.shields.io/badge/SLM--Mode-DBC8FF)

**Experiment guide:** Watch Turing patterns emerge from a Gray–Scott reaction–diffusion system. The activator (U) and inhibitor (V) form spots, stripes, or waves depending on feed/kill parameters (F, K). Use SLM masks to locally perturb concentrations and seed new pattern domains. SLM-Mode toggles between adding activator and adding inhibitor.

**Devices:** SLM — Locally perturbs species concentrations in illuminated regions | SLM-Mode — Toggles perturbation target between activator (U) and inhibitor (V)

**Parameters:** `grid_size` (Simulation grid size in pixels (default 512)) · `preset` (Pattern preset: 'waves', 'spots', 'stripes', etc. (default 'waves')) · `F` (Feed rate (default depends on preset)) · `K` (Kill rate (default depends on preset))

## spheroid

*Multicellular tumour spheroid (MTS) — Brightfield + epifluorescence*

Simulates a multicellular tumour spheroid cross-section with necrotic core and quiescent rim. Features Calcein-AM (live) and propidium-iodide (dead) viability staining.

<p style="display:flex;gap:4px"><img src="gallery/frames/spheroid_01.png" width="24%"> <img src="gallery/frames/spheroid_02.png" width="24%"> <img src="gallery/frames/spheroid_03.png" width="24%"> <img src="gallery/frames/spheroid_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![brightfield](https://img.shields.io/badge/brightfield-ADD3FF) ![Calcein-AM](https://img.shields.io/badge/Calcein--AM-ADD3FF) ![propidium-iodide](https://img.shields.io/badge/propidium--iodide-ADD3FF)

**Experiment guide:** View a cross-section of a multicellular tumour spheroid with a necrotic core, quiescent rim, and proliferating outer shell. Calcein-AM labels live cells green; propidium iodide labels dead cells red. Track spheroid growth and viability gradients over time. Useful for training spheroid segmentation and viability analysis.

**Parameters:** `radius` (Spheroid radius in pixels (default 80)) · `n_cells` (Total number of cells (default 2000)) · `necrotic_fraction` (Fraction of necrotic core (default 0.45))

## spt

*Fluorescent nanoparticles / single molecules — Widefield + TIRF epifluorescence*

Simulates single-particle tracking with free, confined, and directed diffusion modes. Features blinking, bleaching, and widefield/TIRF illumination.

<p style="display:flex;gap:4px"><img src="gallery/frames/spt_01.png" width="24%"> <img src="gallery/frames/spt_02.png" width="24%"> <img src="gallery/frames/spt_03.png" width="24%"> <img src="gallery/frames/spt_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![widefield](https://img.shields.io/badge/widefield-ADD3FF) ![TIRF](https://img.shields.io/badge/TIRF-ADD3FF)

**Experiment guide:** Track single fluorescent particles undergoing free diffusion, confined diffusion, or directed transport. Particles exhibit stochastic blinking and irreversible photobleaching. Compare widefield vs. TIRF illumination for signal-to-noise. Useful for benchmarking single-particle tracking algorithms.

**Parameters:** `n_free` (Free-diffusion particles (default 20)) · `n_confined` (Confined-diffusion particles (default 10)) · `n_directed` (Directed-transport particles (default 5)) · `D_free` (Free diffusion coefficient (default 0.1))

## stress_granule

*Epithelial cells with G3BP1-GFP stress granule reporter — Phase-contrast + epifluorescence*

Simulates stress granule formation and dissolution with G3BP1-GFP foci. Supports SLM stimulation, temperature, and perfusion for stress induction.

<p style="display:flex;gap:4px"><img src="gallery/frames/stress_granule_01.png" width="24%"> <img src="gallery/frames/stress_granule_02.png" width="24%"> <img src="gallery/frames/stress_granule_03.png" width="24%"> <img src="gallery/frames/stress_granule_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![membrane](https://img.shields.io/badge/membrane-ADD3FF) ![SLM](https://img.shields.io/badge/SLM-DBC8FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Perfusion](https://img.shields.io/badge/Perfusion-DBC8FF)

**Experiment guide:** Watch stress granule formation and dissolution in response to cellular stress. G3BP1-GFP foci appear under heat shock, oxidative stress, or SLM-triggered optogenetic stress. Track foci count and size over time. Temperature induces heat-shock stress; perfusion delivers arsenite or other stressors.

**Devices:** SLM — Triggers localised stress response in illuminated cells, inducing granule formation | Temperature — Heat shock (≥43°C) induces rapid stress granule assembly | Perfusion — Delivers chemical stressors (arsenite, thapsigargin) that trigger granule formation

**Parameters:** `n_cells` (Number of cells (default 30)) · `formation_rate` (Stress granule formation rate (default 0.25)) · `dissolution_rate` (Stress granule dissolution rate (default 0.12))

## viability

*Epithelial cells with live/dead viability staining — Phase-contrast + epifluorescence*

Simulates live/dead viability staining with Calcein-AM and Ethidium homodimer-1. Features configurable live fraction with temperature and perfusion.

<p style="display:flex;gap:4px"><img src="gallery/frames/viability_01.png" width="24%"> <img src="gallery/frames/viability_02.png" width="24%"> <img src="gallery/frames/viability_03.png" width="24%"> <img src="gallery/frames/viability_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![membrane](https://img.shields.io/badge/membrane-ADD3FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Perfusion](https://img.shields.io/badge/Perfusion-DBC8FF)

**Experiment guide:** Assess cell viability using Calcein-AM (live, green) and Ethidium homodimer-1 (dead, red) dual staining. A configurable fraction of cells are dead at baseline. Temperature and perfusion can modulate viability over time. Useful for training live/dead classification and cytotoxicity quantification pipelines.

**Devices:** Temperature — Extreme temperatures (≥45°C) induce cell death over time | Perfusion — Delivers cytotoxic agents that reduce viability

**Parameters:** `n_cells` (Number of cells (default 60)) · `live_fraction` (Initial fraction of live cells (default 0.85))

## volvox

*Volvox carteri (colonial green alga) — Brightfield + epifluorescence*

Simulates a swimming Volvox colony with somatic cells and gonidia. Features SLM phototaxis control with chlorophyll and pherophorin channels.

<p style="display:flex;gap:4px"><img src="gallery/frames/volvox_01.png" width="24%"> <img src="gallery/frames/volvox_02.png" width="24%"> <img src="gallery/frames/volvox_03.png" width="24%"> <img src="gallery/frames/volvox_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![brightfield](https://img.shields.io/badge/brightfield-ADD3FF) ![chlorophyll](https://img.shields.io/badge/chlorophyll-ADD3FF) ![pherophorin](https://img.shields.io/badge/pherophorin-ADD3FF) ![SLM](https://img.shields.io/badge/SLM-DBC8FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF)

**Experiment guide:** Watch a Volvox colony swim and rotate. Somatic cells on the surface beat flagella for locomotion; large gonidia inside develop into daughter colonies. Chlorophyll autofluorescence labels all cells; pherophorin marks the extracellular matrix. SLM phototaxis steers the colony toward or away from light.

**Devices:** SLM — Directional light cue that steers phototactic swimming toward illuminated region | Temperature — Affects flagellar beat frequency and swimming speed

**Parameters:** `n_somatic` (Number of somatic cells (default 300)) · `n_gonidia` (Number of reproductive gonidia (default 4)) · `colony_radius` (Colony radius in pixels (default 60)) · `swim_speed` (Swimming speed in px/step (default 3.0))

## voronoi

*Epithelial tissue monolayer (static) — Phase-contrast + epifluorescence*

Simulates a static Voronoi tessellation of an epithelial tissue monolayer. Provides phase-contrast, DAPI nuclear, and membrane channels.

<p style="display:flex;gap:4px"><img src="gallery/frames/voronoi_01.png" width="24%"> <img src="gallery/frames/voronoi_02.png" width="24%"> <img src="gallery/frames/voronoi_03.png" width="24%"> <img src="gallery/frames/voronoi_04.png" width="24%"></p>

![static](https://img.shields.io/badge/static-CBCBCC) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![membrane](https://img.shields.io/badge/membrane-ADD3FF)

**Experiment guide:** A static Voronoi tessellation representing an epithelial monolayer. Navigate the tissue with XY stage and zoom with objectives. Phase-contrast shows cell boundaries, DAPI labels nuclei, and membrane channel highlights cell–cell junctions. Useful as a simple baseline for segmentation benchmarks.

**Parameters:** `n_cells` (Number of cells in the tessellation (default 60)) · `jitter` (Voronoi jitter (randomness) 0–1 (default 0.7)) · `nucleus_fraction` (Nucleus-to-cell area ratio (default 0.3))

## wound_healing

*Epithelial monolayer (scratch-wound assay) — Phase-contrast + epifluorescence*

Simulates a scratch-wound healing assay with migrating epithelial cells. Supports SLM, temperature, perfusion, and electrode for wound closure studies.

<p style="display:flex;gap:4px"><img src="gallery/frames/wound_healing_01.png" width="24%"> <img src="gallery/frames/wound_healing_02.png" width="24%"> <img src="gallery/frames/wound_healing_03.png" width="24%"> <img src="gallery/frames/wound_healing_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![DAPI](https://img.shields.io/badge/DAPI-ADD3FF) ![membrane](https://img.shields.io/badge/membrane-ADD3FF) ![SLM](https://img.shields.io/badge/SLM-DBC8FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Perfusion](https://img.shields.io/badge/Perfusion-DBC8FF) ![Electrode](https://img.shields.io/badge/Electrode-DBC8FF)

**Experiment guide:** Simulates a classic scratch-wound healing assay. Cells are seeded as a confluent monolayer, then a vertical scratch wound is created. Track wound closure over time with phase-contrast time-lapse. Use the Electrode device to apply an electric field for galvanotaxis. Temperature and perfusion control cell migration speed.

**Devices:** SLM — Optogenetic stimulation increases migration speed of illuminated cells | Temperature — Higher temperature increases migration speed | Perfusion — Drug mode can inhibit migration | Electrode — Applies directional electric field for galvanotaxis — cells migrate toward cathode

**Parameters:** `n_cells` (Cell count (default 100)) · `wound_width` (Scratch width in pixels (default 120)) · `migration_speed` (Base migration speed (default 2.0))

## yeast

*Saccharomyces cerevisiae (budding yeast) — Phase-contrast + epifluorescence*

Simulates budding yeast (S. cerevisiae) with cell-wall Calcofluor-White staining and GFP reporter. Supports temperature control and configurable division time.

<p style="display:flex;gap:4px"><img src="gallery/frames/yeast_01.png" width="24%"> <img src="gallery/frames/yeast_02.png" width="24%"> <img src="gallery/frames/yeast_03.png" width="24%"> <img src="gallery/frames/yeast_04.png" width="24%"></p>

![static](https://img.shields.io/badge/static-CBCBCC) ![phase-contrast](https://img.shields.io/badge/phase--contrast-ADD3FF) ![Calcofluor-White](https://img.shields.io/badge/Calcofluor--White-ADD3FF) ![GFP-reporter](https://img.shields.io/badge/GFP--reporter-ADD3FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF)

**Experiment guide:** Observe budding yeast cells with Calcofluor-White cell-wall staining and a GFP reporter. Cells divide by budding at a configurable rate. Temperature affects growth rate — optimal at 30°C, arrested below 4°C or above 42°C. Useful for training budding-index and colony-counting algorithms.

**Devices:** Temperature — Growth rate is temperature-dependent; optimal at 30°C, lethal above 42°C

**Parameters:** `n_cells` (Initial cell count (default 200)) · `world_size` (World size in pixels (default 512)) · `division_time` (Mean division time in steps (default 15.0))

## zebrafish

*Zebrafish embryo (Danio rerio, 48 hpf) — Brightfield + epifluorescence*

Simulates a zebrafish embryo tail with beating heart and circulating RBCs. Features flk1-GFP vasculature and myl7-mCherry cardiac marker with temperature and anesthesia.

<p style="display:flex;gap:4px"><img src="gallery/frames/zebrafish_01.png" width="24%"> <img src="gallery/frames/zebrafish_02.png" width="24%"> <img src="gallery/frames/zebrafish_03.png" width="24%"> <img src="gallery/frames/zebrafish_04.png" width="24%"></p>

![continuous](https://img.shields.io/badge/continuous-AFE2BD) ![brightfield](https://img.shields.io/badge/brightfield-ADD3FF) ![flk1-GFP](https://img.shields.io/badge/flk1--GFP-ADD3FF) ![myl7-mCherry](https://img.shields.io/badge/myl7--mCherry-ADD3FF) ![Temperature](https://img.shields.io/badge/Temperature-DBC8FF) ![Anesthesia](https://img.shields.io/badge/Anesthesia-DBC8FF)

**Experiment guide:** Observe a zebrafish embryo tail region with a beating heart and circulating red blood cells. flk1-GFP labels vasculature endothelium; myl7-mCherry marks cardiac muscle. Temperature affects heart rate and development speed. Anesthesia (tricaine) slows or stops the heartbeat for stable imaging.

**Devices:** Temperature — Heart rate and development speed scale with temperature; optimal at 28°C | Anesthesia — Tricaine (MS-222) progressively slows heartbeat and suppresses movement

**Parameters:** `n_rbc` (Number of circulating red blood cells (default 30)) · `cardiac_freq` (Cardiac beating frequency in Hz (default 2.5))
