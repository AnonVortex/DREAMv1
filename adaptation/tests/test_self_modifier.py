import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from ..adaptation_service import SelfModifier, AdaptationRule, ArchitectureConfig

@pytest.fixture
def self_modifier():
    return SelfModifier()

@pytest.fixture
def sample_architecture_config():
    return ArchitectureConfig(
        components=[
            {
                "name": "perception",
                "type": "core",
                "connections": ["reasoning", "memory"],
                "config": {
                    "batch_size": 32,
                    "model_type": "transformer"
                }
            },
            {
                "name": "reasoning",
                "type": "core",
                "connections": ["memory", "action"],
                "config": {
                    "inference_mode": "fast",
                    "max_depth": 5
                }
            }
        ],
        global_settings={
            "debug_mode": False,
            "monitoring": True
        }
    )

@pytest.fixture
def sample_adaptation_rules():
    return [
        AdaptationRule(
            condition="component.type == 'core' and metrics.latency > 100",
            action="optimize_performance",
            parameters={
                "target": "batch_size",
                "adjustment": -0.2
            }
        ),
        AdaptationRule(
            condition="metrics.memory_usage > 0.8",
            action="scale_resources",
            parameters={
                "component": "memory",
                "factor": 1.5
            }
        )
    ]

def test_apply_adaptation_rule(self_modifier, sample_architecture_config, sample_adaptation_rules):
    rule = sample_adaptation_rules[0]
    metrics = {"latency": 150}  # Trigger the rule
    
    modified_config = self_modifier.apply_adaptation_rule(
        rule,
        sample_architecture_config,
        metrics
    )
    
    # Verify the modification
    perception_component = next(
        c for c in modified_config.components if c["name"] == "perception"
    )
    assert perception_component["config"]["batch_size"] < 32  # Should be reduced

def test_evaluate_condition(self_modifier, sample_adaptation_rules):
    rule = sample_adaptation_rules[0]
    
    # Test condition that should evaluate to True
    metrics = {"latency": 150}
    component = {"type": "core"}
    assert self_modifier.evaluate_condition(rule.condition, component, metrics)
    
    # Test condition that should evaluate to False
    metrics = {"latency": 50}
    assert not self_modifier.evaluate_condition(rule.condition, component, metrics)

def test_modify_architecture(self_modifier, sample_architecture_config):
    modification = {
        "component": "perception",
        "property": "config.batch_size",
        "value": 64
    }
    
    modified_config = self_modifier.modify_architecture(
        sample_architecture_config,
        modification
    )
    
    # Verify the modification
    perception_component = next(
        c for c in modified_config.components if c["name"] == "perception"
    )
    assert perception_component["config"]["batch_size"] == 64

def test_validate_modification(self_modifier, sample_architecture_config):
    # Test valid modification
    valid_modification = {
        "component": "perception",
        "property": "config.batch_size",
        "value": 64
    }
    assert self_modifier.validate_modification(valid_modification, sample_architecture_config)
    
    # Test invalid modification (non-existent component)
    invalid_modification = {
        "component": "nonexistent",
        "property": "config.batch_size",
        "value": 64
    }
    assert not self_modifier.validate_modification(invalid_modification, sample_architecture_config)

def test_check_constraints(self_modifier, sample_architecture_config):
    constraints = {
        "max_connections": 3,
        "required_components": ["perception", "memory"],
        "forbidden_connections": ["perception-action"]
    }
    
    # Test valid configuration
    assert self_modifier.check_constraints(sample_architecture_config, constraints)
    
    # Modify config to violate constraints
    modified_config = ArchitectureConfig(
        components=sample_architecture_config.components + [{
            "name": "invalid",
            "type": "core",
            "connections": ["perception", "reasoning", "memory", "action"],
            "config": {}
        }],
        global_settings=sample_architecture_config.global_settings
    )
    
    assert not self_modifier.check_constraints(modified_config, constraints)

def test_rollback_modification(self_modifier, sample_architecture_config):
    # Make a modification
    modification = {
        "component": "perception",
        "property": "config.batch_size",
        "value": 64
    }
    modified_config = self_modifier.modify_architecture(
        sample_architecture_config,
        modification
    )
    
    # Rollback the modification
    rollback_config = self_modifier.rollback_modification(
        modified_config,
        sample_architecture_config,
        modification
    )
    
    # Verify the rollback
    perception_component = next(
        c for c in rollback_config.components if c["name"] == "perception"
    )
    assert perception_component["config"]["batch_size"] == 32

def test_get_modification_impact(self_modifier, sample_architecture_config):
    modification = {
        "component": "perception",
        "property": "config.batch_size",
        "value": 64
    }
    
    impact = self_modifier.get_modification_impact(modification, sample_architecture_config)
    
    assert isinstance(impact, dict)
    assert "affected_components" in impact
    assert "risk_level" in impact
    assert "estimated_performance_change" in impact

def test_batch_modify_architecture(self_modifier, sample_architecture_config):
    modifications = [
        {
            "component": "perception",
            "property": "config.batch_size",
            "value": 64
        },
        {
            "component": "reasoning",
            "property": "config.max_depth",
            "value": 8
        }
    ]
    
    modified_config = self_modifier.batch_modify_architecture(
        sample_architecture_config,
        modifications
    )
    
    # Verify all modifications
    perception_component = next(
        c for c in modified_config.components if c["name"] == "perception"
    )
    reasoning_component = next(
        c for c in modified_config.components if c["name"] == "reasoning"
    )
    
    assert perception_component["config"]["batch_size"] == 64
    assert reasoning_component["config"]["max_depth"] == 8

def test_get_modification_history(self_modifier):
    history = self_modifier.get_modification_history()
    
    assert isinstance(history, list)
    assert all(isinstance(entry, dict) for entry in history)
    assert all("timestamp" in entry for entry in history)
    assert all("modification" in entry for entry in history)

def test_analyze_architecture_health(self_modifier, sample_architecture_config):
    health_report = self_modifier.analyze_architecture_health(sample_architecture_config)
    
    assert isinstance(health_report, dict)
    assert "overall_health" in health_report
    assert "component_health" in health_report
    assert "recommendations" in health_report

def test_optimize_architecture(self_modifier, sample_architecture_config):
    metrics = {
        "latency": 120,
        "memory_usage": 0.85,
        "cpu_usage": 0.7
    }
    
    optimized_config = self_modifier.optimize_architecture(
        sample_architecture_config,
        metrics
    )
    
    assert isinstance(optimized_config, ArchitectureConfig)
    assert len(optimized_config.components) == len(sample_architecture_config.components)

def test_validate_component_dependencies(self_modifier, sample_architecture_config):
    # Test valid dependencies
    assert self_modifier.validate_component_dependencies(sample_architecture_config)
    
    # Create invalid dependencies (circular)
    invalid_config = ArchitectureConfig(
        components=[
            {
                "name": "component1",
                "type": "core",
                "connections": ["component2"],
                "config": {}
            },
            {
                "name": "component2",
                "type": "core",
                "connections": ["component1"],
                "config": {}
            }
        ],
        global_settings={}
    )
    
    assert not self_modifier.validate_component_dependencies(invalid_config)

def test_generate_modification_plan(self_modifier, sample_architecture_config, sample_adaptation_rules):
    metrics = {
        "latency": 150,
        "memory_usage": 0.9
    }
    
    plan = self_modifier.generate_modification_plan(
        sample_architecture_config,
        metrics,
        sample_adaptation_rules
    )
    
    assert isinstance(plan, list)
    assert all(isinstance(step, dict) for step in plan)
    assert all("modification" in step for step in plan)
    assert all("reason" in step for step in plan) 