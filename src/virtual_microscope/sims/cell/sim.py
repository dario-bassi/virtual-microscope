# Optimized microscope simulation based on the nwe opto-loop implmentation
import numpy as np
from typing import Optional, List, Union, Literal, Sequence
from virtual_microscope.base import SimBase
from virtual_microscope.sims.cell.optogenetic import OptogeneticCell
from virtual_microscope.sims.cell.drug import DrugResponseCell
from virtual_microscope.sims.cell.renderer import CellCycleRenderer
from virtual_microscope.sims.cell.spatial_grid import SpatialGrid
from virtual_microscope.sims.cell.cell import CellBase, update_all_cells_parallel, check_collision
from virtual_microscope.sims.cell.normal import NormalCell
from virtual_microscope.sims.cell.cycle import CellCycleNormal
from virtual_microscope.sims.cell.cycle_manager import CellCycleManager
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline
import cv2


class ScatteredCellSim(SimBase):
    """Optmized microscope simulation with modular cell types."""

    continuous = True

    def __init__(self, width: int = 1500, height: int = 1500,
                 n_cells: int = 240, cell_type: str = "optogenetic",
                 viewport_width: int = 512, viewport_height: int = 512,
                 base_radius: float = 20.0, seed: int = 0,
                 cell_mix: Optional[dict] = None, concentration: float = 0.01, drug_type: Literal["growth", "mobility", "apoptosis"] = "growth"):
        super().__init__(
            width=width, height=height,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=1,
            fixed_dt=0.005,
            mode_map={
                ("SCFP2(434/474)", "UV"): 1,           # DAPI
                ("mScarlet3(569/582)", "ORANGE"): 2,   # membrane
            },
        )

        self.n_cells = n_cells
        self.base_radius = base_radius
        self.cell_type = cell_type
        self.cell_mix = cell_mix
        self.concentration = concentration
        self.drug_type = drug_type
        # Initialize components
        self.renderer = CellCycleRenderer(viewport_width, viewport_height)
        self.spatial_grid = SpatialGrid(width, height, base_radius * 3)

        # Per-channel optical pipelines (noise, PSF, vignette)
        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.4, noise={"photon_scale": 8.0, "read_std": 2.0},
                vignette=0.08, rng_seed=seed + 300),
            1: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 5.0, "read_std": 2.5},
                vignette=0.10, rng_seed=seed + 301),
            2: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 5.0, "read_std": 2.5},
                vignette=0.10, rng_seed=seed + 302),
        }

        # Initialize cells based on type
        self._rng = np.random.RandomState(seed)

        # Create cell objects
        self._cells = self._create_cells()
        self._init_numpy_arrays()

        # Initialize cell cycle manager if using cell cycle cells
        self.cycle_manager: Optional[CellCycleManager] = None
        if self.cell_type == "cycle":
            self.cycle_manager = CellCycleManager(max_divisions=10, track_stats=True)


    def _create_cells(self) -> List[Union[OptogeneticCell, DrugResponseCell, NormalCell, CellCycleNormal]]:
        """Create cells of specific type."""
        cells = []
        if self.cell_mix is None:

            for i in range(self.n_cells):
                seed = self._rng.randint(0, 10000)

                if self.cell_type == "optogenetic":
                    cell = OptogeneticCell(self.width, self.height, self.base_radius,vertices=24, seed=seed) # to check

                elif self.cell_type == "drug":
                    cell = DrugResponseCell(self.width, self.height, self.base_radius, vertices=24, seed=seed) # to check
                elif self.cell_type == "normal":
                    cell = NormalCell(self.width, self.height, self.base_radius, vertices = 24, seed= seed)
                elif self.cell_type == "mixed":
                    # Default mixed population
                        cell_choice = self._rng.choice(['normal', 'optogenetic'], p=[0.7, 0.3])
                        if cell_choice == 'normal':
                            cell = NormalCell(
                                self.width, self.height, self.base_radius,
                                vertices=24, seed=seed
                            )
                        else:
                            cell = OptogeneticCell(
                                self.width, self.height, self.base_radius,
                                vertices=24, seed=seed
                            )
                elif self.cell_type == "cycle":
                    cell = CellCycleNormal(self.width, self.height, self.base_radius, vertices=24, seed=seed, initial_state='M', initial_mitosis='Metaphase', initial_divisions=10, initial_time=496)
                else:
                    raise ValueError(f"Unknow celly type: {self.cell_type}")
                
                cells.append(cell)
        else:
            # cell_mix = {"normal": 60, "optogenetic": 40, "drug": 20}
            #total = sum(self.cell_mix.values())
            for cell_type, count in self.cell_mix.items():
                for _ in range(count):
                    seed = self._rng.randint(0, 10000)
                    
                    if cell_type == "normal":
                        cell = NormalCell(
                            self.width, self.height, self.base_radius,
                            vertices=24, seed=seed
                        )
                    elif cell_type == "optogenetic":
                        cell = OptogeneticCell(
                            self.width, self.height, self.base_radius,
                            vertices=24, seed=seed
                        )
                    elif cell_type == "drug":
                        cell = DrugResponseCell(
                            self.width, self.height, self.base_radius,
                            vertices=24, seed=seed
                        )
                    else:
                        continue
                    
                    cells.append(cell)
            
            # Shuffle cells for random positioning
            self._rng.shuffle(cells)
            #self.n_cells = len(cells)

        return cells
    
    def _init_numpy_arrays(self):
        """Initialize numpy arrays from cell objects for fast physics."""
        self.centers = np.zeros((self.n_cells, 2), dtype=np.float64)
        self.velocities = np.zeros((self.n_cells, 2), dtype=np.float64)
        self.radii = np.zeros((self.n_cells, 24), dtype=np.float64)
        self.base_radii = np.zeros(self.n_cells, dtype=np.float64)
        self.areas = np.zeros(self.n_cells, dtype=np.float64)
        self.angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        
        # Copy data from cell objects to arrays
        for i, cell in enumerate(self._cells):
            self.centers[i] = cell.center
            self.velocities[i] = cell.vel
            self.radii[i] = cell.r
            self.base_radii[i] = cell.base_r
            self.areas[i] = cell.area0

    def _sync_arrays_to_cells(self):
        """Sync numpy arrays back to cell objects."""
        for i, cell in enumerate(self._cells):
            cell.center = self.centers[i].copy()
            cell.vel = self.velocities[i].copy()
            cell.r = self.radii[i].copy()
    
    def _resync_arrays_after_division(self):
        """Rebuild numpy arrays when cell count changes due to divisions/apoptosis."""
        n_cells = len(self._cells)
        
        # Only rebuild if size changed
        if n_cells != len(self.centers):
            # Create new arrays with new size
            self.centers = np.zeros((n_cells, 2), dtype=np.float64)
            self.velocities = np.zeros((n_cells, 2), dtype=np.float64)
            self.radii = np.zeros((n_cells, 24), dtype=np.float64)
            self.base_radii = np.zeros(n_cells, dtype=np.float64)
            self.areas = np.zeros(n_cells, dtype=np.float64)
            
            # Copy data from all cells
            for i, cell in enumerate(self._cells):
                self.centers[i] = cell.center
                self.velocities[i] = cell.vel
                self.radii[i] = cell.r
                self.base_radii[i] = cell.base_r
                self.areas[i] = cell.area0
            
            # Update tracking variable
            self.n_cells = n_cells
    

    def step(self, dt: float = 0.005) -> None:
        """Advance physics by *dt*.  Called by RealtimeEngine or manually."""
        update_all_cells_parallel(
            self.centers, self.velocities, self.radii, self.angles,
            self.base_radii, self.areas, self.width, self.height, dt
        )

        # Sync back to cell objects
        self._sync_arrays_to_cells()

        # Call cell-specific behaviours (safe - no list modification during iteration)
        for cell in self._cells:
            if hasattr(cell, "update_behavior"):
                cell.update_behavior(dt) # type: ignore

        # Handle collisions
        self._handle_collisions_with_spatial_grid()

        # Update cell cycle dynamics (divisions, apoptosis)
        # This modifies self._cells (adds/removes cells), so do it last
        if self.cycle_manager is not None:
            self.cycle_manager.update(self._cells, self) # type: ignore
            # Resync numpy arrays after cell list changes
            # This protects against index misalignment by rebuilding arrays completely
            self._resync_arrays_after_division()

    def _handle_collisions_with_spatial_grid(self) -> None:
        """Use SpatialGrid for collision detection."""
        self.spatial_grid.clear()
        
        # Add all cells to spatial grid
        for i, cell in enumerate(self._cells):
            self.spatial_grid.add_cell(cell, i)
        
        # Check collisions only for nearby cells
        checked_pairs = set()
        for i, cell in enumerate(self._cells):
            potential = self.spatial_grid.get_potential_collisions(cell, i)
            
            for j in potential:
                if (i, j) not in checked_pairs and (j, i) not in checked_pairs:
                    if cell.check_collision(self._cells[j]):
                        # Additional collision response for normal cells
                        if isinstance(cell, NormalCell):
                            cell.respond_to_collision(self._cells[j])
                        #if isinstance(self._cells[j], NormalCell):
                        #    self._cells[j].respond_to_collision(cell)
                    
                    checked_pairs.add((i, j))

    def apply_optogenetic_stimulator(self, mask: np.ndarray) -> None:
        """Apply optogenetic stimulation to cells.

        The mask is in viewport/camera space.  Cell vertex positions are
        converted from world to viewport coordinates using camera_offset
        so the mask always corresponds to what the camera sees.
        """
        if self.cell_type != "optogenetic":
            return

        offset = tuple(self.camera_offset)

        for cell in self._cells:
            if isinstance(cell, OptogeneticCell):
                cell.stimulate(mask, camera_offset=offset)

        # Sync changes back to arrays
        for i, cell in enumerate(self._cells):
            self.radii[i] = cell.r
            self.velocities[i] = cell.vel


    def apply_drug(self, concentration: float, drug_type: str = "growth") -> None:
        """Apply drug to all cells"""
        if self.cell_type != "drug":
            return
        
        for cell in self._cells:
            if isinstance(cell, DrugResponseCell):
                cell.apply_drug(concentration, drug_type)
        
        # Sync changes back to arrays
        for i, cell in enumerate(self._cells):
            self.base_radii[i] = cell.base_r
            self.radii[i] = cell.r
            self.areas[i] = cell.area0

    
    def snap_frame(self, mask: Optional[np.ndarray] = None, intensity: float = 1.0, exposure = 1.0) -> np.ndarray:
        """Capture a frame with optional stimulation.

        Physics is handled by ``step()`` (driven by RealtimeEngine or called
        manually).  ``snap_frame()`` only renders the current state and
        optionally applies an SLM mask.
        """
        # Apply optogentic stimulation if mask is provided
        if mask is not None and self.cell_type == "optogenetic":
            self.apply_optogenetic_stimulator(mask)
        elif mask is not None and self.cell_type == "drug":
            if np.any(mask): # Only if mask has True values
                self.apply_drug(self.concentration, self.drug_type)

        # determine rendering mode from state devices
        self._update_mode()

        # determine objective render from state devices
        self._update_objectif()

        # Update cell fluorescence based on mode
        for cell in self._cells:
            self._update_cell_fluorescence(cell, self.mode)

        # Render frame based on cell type
        if self.cell_type == "cycle":
            img = self.renderer.render_cell_cycle(
                self._cells, self.mode,  # type: ignore
                tuple(self.camera_offset),
                self.focal_plane)
        else:
            img = self.renderer.render_cells(
                self._cells, self.mode, # type: ignore
                tuple(self.camera_offset),
                self.focal_plane)

        # Apply optical pipeline (noise, PSF, vignette)
        img = self._apply_pipeline(img, exposure)

        # Apply intensity and exposure
        # Brightness is normalized: 1.0 = normal brightness (scaled by 0.01 internally)
        img = (img.astype(np.float32) * intensity * 0.01 * exposure).clip(0,255).astype(np.uint8)

        # return grayscale for compatibility
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def _on_objective_changed(self, mag: int, dof: float) -> None:
        """Forward objective changes to the cell renderer."""
        self.renderer.set_objective(mag, dof)
            

    def _update_cell_fluorescence(self, cell: CellBase, mode: int) -> None:
        """Update cell fluorescence base on mode"""
        if isinstance(cell, NormalCell):
            if mode == 0:
                # Brightfield - no fluorescence visible
                pass # Keep markes ans skip
            elif mode == 1:
                # Shows nucleus only if cell has marker
                if not cell.has_nucleus_marker:
                    cell.nucleus_fluorescence = 0.0
            elif mode == 2:
                # Show membrane only if cell has marker
                if not cell.has_membrane_marker:
                    cell.membrane_fluorescence[:] = 0.0
        
        # For other types of cell (optogenetic/drug), always show fluorescence
        if mode == 0:
            cell.nucleus_fluorescence = 0.0
            cell.membrane_fluorescence[:] = 0.0
        elif mode == 1:
            cell.nucleus_fluorescence = 1.0
        elif mode == 2:
            cell.membrane_fluorescence[:] = 1.0


    def get_visible_cells(self) -> Sequence[CellBase | CellCycleNormal]:
        """Get cells visible in current viewport."""
        return self.renderer._get_visible_cells(self._cells, tuple(self.camera_offset))
    
    def reset(self) -> None:
        """Reset simulation."""
        self._cells = self._create_cells()
        self._init_numpy_arrays()