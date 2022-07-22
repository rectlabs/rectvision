from recognize import recognize_pdf
import unittest

file = 'test_data/abc.pdf'
class TestPDFTextRecognizer(unittest.TestCase):
    def test_recognize_pdf(self):
        self.assertEqual(recognize_pdf(file), "ABC\n")
    

if __name__ == "__main__":
    unittest.main()
