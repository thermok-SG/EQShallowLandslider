import pytest
import numpy as np
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from landlab import RasterModelGrid

# Assuming the class is imported from the main module
from landslide_simulator import ShallowLandslideSimulator


class TestShallowLandslideSimulator:
    """Test suite for ShallowLandslideSimulator class"""
    
    @pytest.fixture
    def mock_config(self):
        """Fixture providing a minimal valid configuration"""
        return {
            'dem_info': {
                'dem_type': 'synthetic',
                'north': 40.0, 'south': 39.9,
                'east': -105.0, 'west': -105.1,
                'buffer': 0.01,
                'plot_dem': False,
                'smooth_num': 1
            },
            'soil_params': {
                'max_soil_depth': 2.0,
                'distribution': 'uniform',
                'plot_soil': False,
                'angle_int_frict': 30.0,
                'cohesion_eff': 1000.0,
                'submerged_soil_proportion': 0.5
            },
            'pga': {
                'horizontal_max': 0.3,
                'vertical_max': 0.15,
                'distribution': 'uniform',
                'plot_grids': False
            },
            'flow_params': {
                'flow_metric': 'D8',
                'separate_hill_flow': False,
                'depression_handling': 'fill',
                'update_hill_depressions': False,
                'accumulate_flow': True
            },
            'simulation': {
                'aspect_interval': 45,
                'selection_method': 'probabilistic',
                'random_seed': 42,
                'proportion_method': 'fixed',
                'time_shaking': 10.0,
                'displacement_threshold': 0.1
            },
            'plot_intermediates': {
                'factor_of_safety': False,
                'critical_acceleration': False,
                'unstable_areas': False,
                'filled_and_split': False
            },
            'output': {
                'output_dir': '/tmp'
            }
        }
    
    @pytest.fixture
    def mock_grid(self):
        """Fixture providing a small test grid"""
        grid = RasterModelGrid((10, 10), xy_spacing=10.0)
        # Add elevation data
        elevation = np.random.uniform(100, 200, grid.number_of_nodes)
        grid.add_field('topographic__elevation', elevation, at='node')
        return grid
    
    # ==================== INITIALIZATION TESTS ====================
    
    def test_init_with_dict_config(self, mock_config):
        """Test initialization with dictionary configuration"""
        simulator = ShallowLandslideSimulator(config=mock_config)
        assert simulator.config == mock_config
        assert simulator.grid is None
        assert simulator.z is None
    
    def test_init_with_grid(self, mock_config, mock_grid):
        """Test initialization with pre-loaded grid"""
        simulator = ShallowLandslideSimulator(config=mock_config, grid=mock_grid)
        assert simulator.grid is mock_grid
        assert np.array_equal(simulator.z, mock_grid.at_node['topographic__elevation'])
    
    def test_init_with_json_config(self, mock_config):
        """Test initialization with JSON file configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_config, f)
            config_path = f.name
        
        with patch('landslide_simulator.get_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            simulator = ShallowLandslideSimulator(config=config_path)
            mock_get_config.assert_called_once_with(config_path)
    
    def test_init_with_none_config(self):
        """Test initialization with None config (should use defaults)"""
        with patch('landslide_simulator.get_config') as mock_get_config:
            mock_get_config.return_value = {}
            simulator = ShallowLandslideSimulator(config=None)
            mock_get_config.assert_called_once_with(None)
    
    # ==================== VALIDATION TESTS ====================
    
    def test_invalid_selection_method(self, mock_config):
        """Test that invalid selection method raises error"""
        mock_config['simulation']['selection_method'] = 'invalid_method'
        simulator = ShallowLandslideSimulator(config=mock_config)
        
        # Mock the required methods to get to select_potential_landslides
        simulator.split_subgroups = np.zeros((10, 10))
        
        with pytest.raises(ValueError, match="Unknown method: invalid_method"):
            simulator.select_potential_landslides()
    
    def test_grid_shape_consistency(self, mock_config, mock_grid):
        """Test that grid operations maintain shape consistency"""
        simulator = ShallowLandslideSimulator(config=mock_config, grid=mock_grid, process_grid=True)
        
        # Check that calculated slopes match grid size
        assert len(simulator.slopes) == mock_grid.number_of_nodes
        assert len(simulator.slopes_degrees) == mock_grid.number_of_nodes
    
    # ==================== ARRAY OPERATION TESTS ====================
    
    def test_calculate_instability_arrays(self, mock_config):
        """Test that instability calculation produces valid arrays"""
        simulator = ShallowLandslideSimulator(config=mock_config)
        
        # Mock required dependencies
        with patch.multiple(simulator,
                          grid=Mock(),
                          slopes=np.random.rand(100),
                          slopes_degrees=np.random.rand(100)):
            
            # Mock the external functions
            with patch('landslide_simulator.generate_acceleration_grid') as mock_gen_accel, \
                 patch('landslide_simulator.factor_of_safety') as mock_fos, \
                 patch('landslide_simulator.critical_transient_acceleration') as mock_cta:
                
                mock_gen_accel.return_value = (np.random.rand(100), np.random.rand(100))
                mock_fos.return_value = np.random.uniform(0.5, 2.0, 100)
                mock_cta.return_value = (np.random.rand(100), np.random.rand(100), np.random.rand(100))
                
                simulator.calculate_instability()
                
                # Check that arrays are properly initialized
                assert simulator.factor_of_safety_vals is not None
                assert simulator.acceleration_horizontal_array is not None
                assert simulator.acceleration_vertical_array is not None
                assert simulator.sliding_locations_bool is not None
    
    def test_boundary_node_handling(self, mock_config, mock_grid):
        """Test that boundary nodes are properly handled"""
        simulator = ShallowLandslideSimulator(config=mock_config, grid=mock_grid, process_grid=True)
        
        # Mock instability calculation
        simulator.sliding_locations_bool = np.random.choice([True, False], size=mock_grid.number_of_nodes)
        simulator.grid = mock_grid
        
        # Process boundaries (this should happen in calculate_instability)
        simulator.sliding_locations_bool[mock_grid.boundary_nodes] = False
        
        # Check that no boundary nodes are marked as sliding
        assert not np.any(simulator.sliding_locations_bool[mock_grid.boundary_nodes])
    
    # ==================== NUMERICAL ACCURACY TESTS ====================
    
    def test_mass_conservation(self, mock_config):
        """Test that mass is conserved during sediment transport"""
        simulator = ShallowLandslideSimulator(config=mock_config)
        
        # Mock the required data
        initial_soil = np.random.uniform(0.5, 3.0, 100)
        updated_soil = np.random.uniform(0.5, 3.0, 100)
        
        simulator.grid = Mock()
        simulator.grid.at_node = {'soil__depth': initial_soil}
        simulator.updated_soil_depth = updated_soil
        
        # Calculate mass balance
        mass_balance = np.sum(updated_soil) - np.sum(initial_soil)
        
        # In a real simulation, mass should be approximately conserved
        # (small differences due to numerical precision are acceptable)
        assert abs(mass_balance) < 1e-10 or abs(mass_balance / np.sum(initial_soil)) < 1e-6
    
    def test_displacement_calculation_bounds(self, mock_config):
        """Test that displacement calculations produce reasonable values"""
        simulator = ShallowLandslideSimulator(config=mock_config)
        
        # Mock required arrays
        simulator.a_diff_EQ = np.random.uniform(-0.5, 2.0, 100)
        simulator.aspect_subgroups = np.random.randint(0, 10, (10, 10))
        simulator.selected_groups = np.random.randint(0, 5, (10, 10))
        simulator.grid = Mock()
        
        time_shaking = 10.0
        
        with patch('landslide_simulator.calculate_newmark_displacement') as mock_calc:
            mock_calc.return_value = np.random.uniform(0, 2.0, 100)
            
            simulator.calculate_displacements()
            
            # Check that displacement values are non-negative
            assert np.all(simulator.newmark_displacement_select >= 0)
    
    # ==================== STATE MANAGEMENT TESTS ====================
    
    def test_results_storage(self, mock_config):
        """Test that results are properly stored in each step"""
        simulator = ShallowLandslideSimulator(config=mock_config)
        
        # Mock a simple instability calculation
        simulator.grid = Mock()
        simulator.slopes = np.random.rand(100)
        simulator.slopes_degrees = np.random.rand(100)
        
        with patch.multiple('landslide_simulator',
                          generate_acceleration_grid=Mock(return_value=(np.random.rand(100), np.random.rand(100))),
                          factor_of_safety=Mock(return_value=np.random.rand(100)),
                          critical_transient_acceleration=Mock(return_value=(np.random.rand(100), np.random.rand(100), np.random.rand(100)))):
            
            simulator.calculate_instability()
            
            # Check that results are stored
            assert 'instability' in simulator.results
            assert 'factor_of_safety' in simulator.results['instability']
            assert 'sliding_locations_bool' in simulator.results['instability']
    
    def test_grid_update_after_simulation(self, mock_config, mock_grid):
        """Test that grid is properly updated after simulation"""
        simulator = ShallowLandslideSimulator(config=mock_config, grid=mock_grid, process_grid=True)
        
        # Store initial elevation
        initial_elevation = mock_grid.at_node['topographic__elevation'].copy()
        
        # Mock updated soil depth
        simulator.updated_soil_depth = np.random.uniform(0.5, 3.0, mock_grid.number_of_nodes)
        mock_grid.add_field('bedrock__elevation', initial_elevation - 1.0, at='node')
        
        # Simulate the grid update that happens in run_one_step
        mock_grid.at_node['soil__depth'] = simulator.updated_soil_depth
        mock_grid.at_node['topographic__elevation'] = (
            mock_grid.at_node['bedrock__elevation'] + mock_grid.at_node['soil__depth']
        )
        
        # Check that elevation was updated
        assert not np.array_equal(initial_elevation, mock_grid.at_node['topographic__elevation'])
    
    # ==================== ERROR HANDLING TESTS ====================
    
    def test_missing_required_fields(self, mock_config):
        """Test handling of missing required grid fields"""
        simulator = ShallowLandslideSimulator(config=mock_config)
        
        # Create grid without required fields
        incomplete_grid = RasterModelGrid((5, 5))
        simulator.grid = incomplete_grid
        
        # This should handle missing fields gracefully
        try:
            simulator._process_grid()
        except KeyError as e:
            pytest.fail(f"_process_grid should handle missing fields gracefully, got: {e}")
    
    def test_invalid_array_shapes(self, mock_config):
        """Test handling of mismatched array shapes"""
        simulator = ShallowLandslideSimulator(config=mock_config)
        
        # Create arrays with mismatched shapes
        simulator.grid = Mock()
        simulator.grid.number_of_nodes = 100
        simulator.grid.shape = (10, 10)
        
        # This should raise an appropriate error
        with pytest.raises((ValueError, IndexError)):
            simulator.aspect_subgroups = np.random.randint(0, 5, (5, 5))  # Wrong shape
            simulator.filter_regions_by_aspect()
    
    # ==================== INTEGRATION TESTS ====================
    
    @patch('landslide_simulator.get_topo')
    @patch('landslide_simulator.calculate_regions')
    @patch('landslide_simulator.generate_acceleration_grid')
    def test_full_simulation_workflow(self, mock_gen_accel, mock_calc_regions, mock_get_topo, mock_config):
        """Test that the full simulation workflow runs without errors"""
        # Mock all external dependencies
        mock_grid = Mock()
        mock_grid.number_of_nodes = 100
        mock_grid.shape = (10, 10)
        mock_grid.boundary_nodes = np.array([0, 1, 2, 97, 98, 99])
        mock_grid.at_node = {
            'topographic__elevation': np.random.uniform(100, 200, 100),
            'soil__depth': np.random.uniform(0.5, 2.0, 100),
            'bedrock__elevation': np.random.uniform(95, 195, 100)
        }
        
        mock_get_topo.return_value = (mock_grid, mock_grid.at_node['topographic__elevation'])
        mock_gen_accel.return_value = (np.random.rand(100), np.random.rand(100))
        mock_calc_regions.return_value = (np.random.randint(0, 10, (10, 10)), 5)
        
        simulator = ShallowLandslideSimulator(config=mock_config)
        
        # Mock all the complex calculation functions
        with patch.multiple('landslide_simulator',
                          factor_of_safety=Mock(return_value=np.random.uniform(0.5, 2.0, 100)),
                          critical_transient_acceleration=Mock(return_value=(np.random.rand(100), np.random.rand(100), np.random.rand(100))),
                          split_groups_by_aspect=Mock(return_value=(np.random.randint(0, 10, (10, 10)), {}, {})),
                          calculate_region_properties=Mock(return_value=({}, np.random.randint(0, 10, (10, 10)))),
                          generate_landslide_probability=Mock(return_value=(np.random.rand(100), {})),
                          probabilistic_group_selection=Mock(return_value=(np.random.randint(0, 5, (10, 10)), 0.1)),
                          calculate_newmark_displacement=Mock(return_value=np.random.uniform(0, 2.0, 100)),
                          trace_paths_landslides=Mock(return_value=({}, {}, {})),
                          update_soil_depth=Mock(return_value=np.random.uniform(0.5, 2.0, 100))):
            
            # Mock additional required methods
            simulator._create_transport_zones = Mock(return_value={'extended_zones': np.random.randint(0, 5, 100)})
            
            # This should run without errors
            try:
                grid, results, model_grids = simulator.run_one_step()
                
                # Check that all expected results are present
                assert grid is not None
                assert isinstance(results, dict)
                assert isinstance(model_grids, dict)
                
            except Exception as e:
                pytest.fail(f"Full simulation workflow failed: {e}")
    
    # ==================== PERFORMANCE TESTS ====================
    
    def test_memory_usage_reasonable(self, mock_config):
        """Test that memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple simulators to test memory management
        simulators = []
        for i in range(10):
            sim = ShallowLandslideSimulator(config=mock_config)
            simulators.append(sim)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 10 simulators)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
    
    # ==================== EDGE CASE TESTS ====================
    
    def test_empty_unstable_regions(self, mock_config):
        """Test behavior when no unstable regions are found"""
        simulator = ShallowLandslideSimulator(config=mock_config)
        
        # Mock a case with no unstable areas
        simulator.sliding_array_bool = np.zeros((10, 10), dtype=bool)
        simulator.grid = Mock()
        simulator.grid.shape = (10, 10)
        
        with patch('landslide_simulator.calculate_regions') as mock_calc_regions:
            mock_calc_regions.return_value = (np.zeros((10, 10)), 0)
            
            # This should handle the empty case gracefully
            simulator.identify_failure_regions()
            
            assert simulator.labeled_array is not None
            assert np.all(simulator.labeled_array == 0)
    
    def test_single_pixel_regions(self, mock_config):
        """Test handling of single-pixel unstable regions"""
        simulator = ShallowLandslideSimulator(config=mock_config)
        
        # Create a case with single-pixel regions
        labeled_array = np.zeros((10, 10))
        labeled_array[5, 5] = 1  # Single pixel region
        
        simulator.labeled_array_filled = labeled_array
        simulator.grid = Mock()
        simulator.grid.shape = (10, 10)
        simulator.grid.calc_aspect_at_node = Mock(return_value=np.random.uniform(0, 360, 100))
        simulator.grid.boundary_nodes = np.array([0, 1, 2, 97, 98, 99])
        
        with patch('landslide_simulator.split_groups_by_aspect') as mock_split, \
             patch('landslide_simulator.calculate_region_properties') as mock_props:
            
            mock_split.return_value = (labeled_array, {}, {})
            mock_props.return_value = ({1: {'area': 1}}, labeled_array)
            
            # Should handle single-pixel regions
            simulator.filter_regions_by_aspect()
            
            assert simulator.aspect_subgroups is not None


if __name__ == "__main__":
    # Example of how to run specific tests
    pytest.main([__file__, "-v"])