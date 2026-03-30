"""Tests for ROS2 package export."""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET

import pytest

from worldkit.core.config import get_config
from worldkit.core.model import WorldModel
from worldkit.export.ros2_export import export_ros2


@pytest.fixture
def model():
    """Create a minimal WorldModel for testing."""
    from worldkit.core.backends import backend_registry

    config = get_config("nano", action_dim=2)
    backend_cls = backend_registry.get(config.backend)
    backend = backend_cls()
    module = backend.build(config)
    return WorldModel(module, config, device="cpu", backend=backend)


@pytest.fixture
def ros2_pkg(tmp_path, model):
    """Export a ROS2 package and return its path."""
    output_dir = tmp_path / "ros2_ws" / "src" / "worldkit_node"
    export_ros2(model, output_dir=output_dir, node_name="worldkit_node")
    return output_dir


def test_ros2_export_generates_package(ros2_pkg):
    """Check all expected files exist in the generated package."""
    expected_files = [
        "package.xml",
        "setup.py",
        "setup.cfg",
        "resource/worldkit_node",
        "worldkit_node/__init__.py",
        "worldkit_node/worldkit_node.py",
        "launch/worldkit.launch.py",
        "config/params.yaml",
        "models/model.wk",
    ]
    for rel_path in expected_files:
        assert (ros2_pkg / rel_path).exists(), f"Missing: {rel_path}"


def test_ros2_node_is_valid_python(ros2_pkg):
    """The generated node file must be parseable Python."""
    node_file = ros2_pkg / "worldkit_node" / "worldkit_node.py"
    source = node_file.read_text()
    tree = ast.parse(source)

    # Verify key class and function exist
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert "WorldKitNode" in names
    assert "main" in names


def test_ros2_package_xml_valid(ros2_pkg):
    """The generated package.xml must be valid XML with required elements."""
    xml_path = ros2_pkg / "package.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()

    assert root.tag == "package"
    assert root.find("name").text == "worldkit_node"
    assert root.find("version") is not None
    assert root.find("description") is not None
    assert root.find("license") is not None

    # Check required ROS2 dependencies
    dep_names = {dep.text for dep in root.findall("depend")}
    assert "rclpy" in dep_names
    assert "sensor_msgs" in dep_names

    # Check build type
    export_el = root.find("export")
    assert export_el is not None
    build_type = export_el.find("build_type")
    assert build_type is not None
    assert build_type.text == "ament_python"


def test_ros2_launch_file_valid(ros2_pkg):
    """The generated launch file must be parseable Python with generate_launch_description."""
    launch_file = ros2_pkg / "launch" / "worldkit.launch.py"
    source = launch_file.read_text()
    tree = ast.parse(source)

    func_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "generate_launch_description" in func_names


def test_ros2_export_custom_node_name(tmp_path, model):
    """Export with a custom node name uses that name throughout."""
    output_dir = tmp_path / "custom_pkg"
    export_ros2(model, output_dir=output_dir, node_name="my_robot_wm")

    assert (output_dir / "my_robot_wm" / "worldkit_node.py").exists()
    assert (output_dir / "resource" / "my_robot_wm").exists()

    xml_tree = ET.parse(output_dir / "package.xml")
    assert xml_tree.getroot().find("name").text == "my_robot_wm"

    setup_py = (output_dir / "setup.py").read_text()
    assert 'package_name = "my_robot_wm"' in setup_py
