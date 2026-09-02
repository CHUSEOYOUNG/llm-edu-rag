from pathlib import Path
import tomllib
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ContainerConfigTests(unittest.TestCase):
    def test_runtime_pins_torch_to_the_cpu_only_index(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(project["tool"]["uv"]["sources"]["torch"],
                         {"index": "pytorch-cpu"})
        indexes = {item["name"]: item for item in project["tool"]["uv"]["index"]}
        self.assertEqual(indexes["pytorch-cpu"]["url"],
                         "https://download.pytorch.org/whl/cpu")
        self.assertTrue(indexes["pytorch-cpu"]["explicit"])

    def test_image_uses_locked_runtime_dependencies_and_non_root_user(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("uv sync --locked --no-dev --no-install-project", dockerfile)
        self.assertIn("USER app", dockerfile)
        self.assertIn('HF_HUB_OFFLINE=1', dockerfile)
        self.assertIn('TRANSFORMERS_OFFLINE=1', dockerfile)
        self.assertIn('HEALTHCHECK', dockerfile)
        self.assertIn('"uvicorn", "search_app:app"', dockerfile)
        self.assertIn('"--host", "0.0.0.0"', dockerfile)

    def test_compose_keeps_public_port_and_model_inputs_local(self):
        compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
        search = compose["services"]["search"]
        self.assertEqual(search["ports"], ["127.0.0.1:8765:8765"])
        mounts = {mount["target"]: mount for mount in search["volumes"]}
        for target in ("/app/data/processed", "/app/data/raw",
                       "/home/app/.cache/huggingface"):
            self.assertTrue(mounts[target]["read_only"])
        self.assertTrue(search["read_only"])
        self.assertEqual(search["cap_drop"], ["ALL"])
        self.assertEqual(search["security_opt"], ["no-new-privileges:true"])

    def test_large_or_private_runtime_files_stay_out_of_build_context(self):
        ignored = set((ROOT / ".dockerignore").read_text().splitlines())
        for path in (".git", ".venv", ".env", "data", "runs", "tests"):
            self.assertIn(path, ignored)


if __name__ == "__main__":
    unittest.main()
