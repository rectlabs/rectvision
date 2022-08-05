import unittest
import subprocess

class TestPDFTextRecognizer(unittest.TestCase):
    def test_installation(self):
        subprocess.run(['bash', 'run.sh'])

    def test_recognize(self):
        result = subprocess.run(['python', 'recognize.py', '--file_path', 'test_data/abc.pdf'])   

if __name__ == "__main__":
    unittest.main()
