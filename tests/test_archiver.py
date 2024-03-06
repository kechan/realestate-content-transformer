import unittest
from realestate_content_transformer.data.archive import ChatGPTRewriteArchiver, ArchiveStorageType
import realestate_core.common.class_extensions
# run to run
# python -m unittest test_archiver.py
# python -m unittest test_archiver.TestChatGPTRewriteArchiver.test_add_and_get_record

class TestChatGPTRewriteArchiver(unittest.TestCase):
  def setUp(self):
    self.archiver = ChatGPTRewriteArchiver(ArchiveStorageType.PLAIN_TEXT, file_path='test_archive.txt')

  def tearDown(self):
    import os
    os.remove('test_archive.txt')

  def test_add_and_get_record(self):
    # Test for lang='en'
    self.archiver.add_record(
        longId='ab_calgary',
        property_type='LUXURY',
        version='202311',
        lang='en',
        user_prompt="No use prompt, this is a test",
        chatgpt_response="Calgary boasts an impressive skyline and lively cultural scene.",
    )
    record = self.archiver.get_record(longId='ab_calgary', property_type='LUXURY', version='202311', lang='en')
    self.assertIsNotNone(record)
    self.assertEqual(record['user_prompt'], "No use prompt, this is a test")
    self.assertEqual(record['chatgpt_response'], "Calgary boasts an impressive skyline and lively cultural scene.")

    # Test for lang='fr'
    self.archiver.add_record(
        longId='ab_calgary',
        property_type='LUXURY',
        version='202311',
        lang='fr',
        user_prompt="Pas d'invite d'utilisation, c'est un test",
        chatgpt_response="Calgary possède une skyline impressionnante et une scène culturelle animée.",
    )
    record = self.archiver.get_record(longId='ab_calgary', property_type='LUXURY', version='202311', lang='fr')
    self.assertIsNotNone(record)
    self.assertEqual(record['user_prompt'], "Pas d'invite d'utilisation, c'est un test")
    self.assertEqual(record['chatgpt_response'], "Calgary possède une skyline impressionnante et une scène culturelle animée.")

  def test_get_nonexistent_record(self):
    self.archiver.add_record(
        longId='ab_calgary',
        property_type='LUXURY',
        version='202311',
        lang='en',
        user_prompt="No use prompt, this is a test",
        chatgpt_response="Calgary boasts an impressive skyline and lively cultural scene.",
    )

    # unmatched longId
    record = self.archiver.get_record('ab_calgary_xyz', 'LUXURY', '202311', 'en')
    self.assertIsNone(record)

    # unmatched property_type
    record = self.archiver.get_record('ab_calgary', 'LUXURY_xyz', '202311', 'en')
    self.assertIsNone(record)

    # unmatched version
    record = self.archiver.get_record('ab_calgary', 'LUXURY', '202311_xyz', 'en')
    self.assertIsNone(record)
  
  def test_get_record_with_version_prefix(self):
    self.archiver.add_record(
        longId='ab_calgary',
        property_type='LUXURY',
        version='202311',
        lang='en',
        user_prompt="No use prompt, this is a test",
        chatgpt_response="Calgary boasts an impressive skyline and lively cultural scene.",
    )
    record = self.archiver.get_record('ab_calgary', 'LUXURY', '2023', 'en')
    self.assertIsNotNone(record)
    self.assertEqual(record['user_prompt'], "No use prompt, this is a test")
    self.assertEqual(record['chatgpt_response'], "Calgary boasts an impressive skyline and lively cultural scene.")

  def test_get_record_from_empty_archive(self):
    empty_archiver = ChatGPTRewriteArchiver(ArchiveStorageType.PLAIN_TEXT, file_path='test_empty_archive.txt')
    record = empty_archiver.get_record('ab_calgary', 'LUXURY', '202311', 'en')
    self.assertIsNone(record)
    # Clean up after the test
    import os
    os.remove('test_empty_archive.txt')

  def test_get_record_with_cache(self):
    # First, create an archive with a record
    initial_archiver = ChatGPTRewriteArchiver(ArchiveStorageType.PLAIN_TEXT, file_path='test_archive.txt')
    initial_archiver.add_record(
        longId='ab_calgary',
        property_type='LUXURY',
        version='202311',
        lang='en',
        user_prompt="No use prompt, this is a test",
        chatgpt_response="Calgary boasts an impressive skyline and lively cultural scene.",
    )
    del initial_archiver  # Explicitly delete the initial archiver to free up the file resource

    # Then, create a new archiver that will load the existing record into its cache
    self.archiver = ChatGPTRewriteArchiver(ArchiveStorageType.PLAIN_TEXT, file_path='test_archive.txt')

    # Now, the record should be in the cache
    record = self.archiver.get_record('ab_calgary', 'LUXURY', '202311', 'en', use_cache=True)
    self.assertIsNotNone(record)
    self.assertEqual(record['user_prompt'], "No use prompt, this is a test")
    self.assertEqual(record['chatgpt_response'], "Calgary boasts an impressive skyline and lively cultural scene.")

  def test_get_nonexistent_record_with_cache(self):
    initial_archiver = ChatGPTRewriteArchiver(ArchiveStorageType.PLAIN_TEXT, file_path='test_archive.txt')
    initial_archiver.add_record(
        longId='ab_calgary',
        property_type='LUXURY',
        version='202311',
        lang='en',
        user_prompt="No use prompt, this is a test",
        chatgpt_response="Calgary boasts an impressive skyline and lively cultural scene.",
    )
    del initial_archiver  # Explicitly delete the initial archiver to free up the file resource

    self.archiver = ChatGPTRewriteArchiver(ArchiveStorageType.PLAIN_TEXT, file_path='test_archive.txt')

    # unmatched longId
    record = self.archiver.get_record('ab_calgary_xyz', 'LUXURY', '202311', 'en', use_cache=True)
    self.assertIsNone(record)

    # unmatched property_type
    record = self.archiver.get_record('ab_calgary', 'LUXURY_xyz', '202311', 'en', use_cache=True)
    self.assertIsNone(record)

    # unmatched version
    record = self.archiver.get_record('ab_calgary', 'LUXURY', '202311_xyz', 'en', use_cache=True)
    self.assertIsNone(record)

  def test_get_record_with_version_prefix_with_cache(self):
    initial_archiver = ChatGPTRewriteArchiver(ArchiveStorageType.PLAIN_TEXT, file_path='test_archive.txt')
    initial_archiver.add_record(
        longId='ab_calgary',
        property_type='LUXURY',
        version='202311',
        lang='en',
        user_prompt="No use prompt, this is a test",
        chatgpt_response="Calgary boasts an impressive skyline and lively cultural scene.",
    )
    del initial_archiver  # Explicitly delete the initial archiver to free up the file resource

    self.archiver = ChatGPTRewriteArchiver(ArchiveStorageType.PLAIN_TEXT, file_path='test_archive.txt')

    record = self.archiver.get_record('ab_calgary', 'LUXURY', '2023', 'en', use_cache=True)
    self.assertIsNotNone(record)
    self.assertEqual(record['user_prompt'], "No use prompt, this is a test")
    self.assertEqual(record['chatgpt_response'], "Calgary boasts an impressive skyline and lively cultural scene.")

  def test_get_record_from_empty_archive_with_cache(self):
    empty_archiver = ChatGPTRewriteArchiver(ArchiveStorageType.PLAIN_TEXT, file_path='test_empty_archive.txt')
    record = empty_archiver.get_record('ab_calgary', 'LUXURY', '202311', 'en', use_cache=True)
    self.assertIsNone(record)
    # Clean up after the test
    import os
    os.remove('test_empty_archive.txt')

  def test_get_all_records(self):
    # Add multiple records to the archive
    records_to_add = [
        {
            'longId': 'ab_calgary',
            'property_type': 'LUXURY',
            'version': '202311',
            'lang': 'en',
            'user_prompt': "No use prompt, this is a test",
            'chatgpt_response': "Calgary boasts an impressive skyline and lively cultural scene."
        },
        {
            'longId': 'bc_vancouver',
            'property_type': 'INVESTMENT',
            'version': '202312',
            'lang': 'en',
            'user_prompt': "No use prompt, this is a test",
            'chatgpt_response': "Vancouver is known for its beautiful scenery and outdoor activities."
        },
        # Add more records as needed
    ]
    for record in records_to_add:
      self.archiver.add_record(**record)

    # Call get_all_records
    records = self.archiver.get_all_records()

    # Check that all the added records are returned
    for record in records_to_add:
      key = f"{record['longId']}:{record['property_type']}:{record['version']}:{record['lang']}"
      self.assertIn(key, records)
      for field in ['user_prompt', 'chatgpt_response']:
        self.assertEqual(record[field], records[key][field])

  def test_add_record_with_invalid_property_type(self):
    with self.assertRaises(ValueError):
      self.archiver.add_record(
        longId='bc_vancouver',
        property_type='INVALID_PROPERTY_TYPE',
        version='202312',
        lang='en',
        user_prompt="No use prompt, this is a test",
        chatgpt_response="Vancouver is known for its beautiful scenery and outdoor activities."
      )

  def test_add_record_with_missing_fields(self):
    with self.assertRaises(TypeError):
      self.archiver.add_record(
          longId='ab_calgary',
          property_type='LUXURY',
          version='202311',
          lang='en'
          # Missing 'user_prompt' and 'chatgpt_response'
      )

  def test_add_record_with_extra_fields(self):
    with self.assertRaises(TypeError):
      self.archiver.add_record(
          longId='ab_calgary',
          property_type='LUXURY',
          version='202311',
          lang='en',
          user_prompt="No use prompt, this is a test",
          chatgpt_response="Calgary boasts an impressive skyline and lively cultural scene.",
          extra_field="This field shouldn't be here"
      )

  def test_add_duplicate_record(self):
    self.archiver.add_record(
        longId='ab_calgary',
        property_type='LUXURY',
        version='202311',
        lang='en',
        user_prompt="No use prompt, this is a test",
        chatgpt_response="Calgary boasts an impressive skyline and lively cultural scene.",
    )
    # Add another record with the same identifiers but different user_prompt and chatgpt_response    
    self.archiver.add_record(
        longId='ab_calgary',
        property_type='LUXURY',
        version='202311',
        lang='en',
        user_prompt="Different prompt",
        chatgpt_response="Different response",
    )

    # Check that get_record returns the new record
    record = self.archiver.get_record('ab_calgary', 'LUXURY', '202311', 'en')
    self.assertEqual(record['user_prompt'], "Different prompt")
    self.assertEqual(record['chatgpt_response'], "Different response")
  
  def test_add_duplicate_record_with_cache(self):
    # Add a record and overwrite it with a new one
    self.archiver.add_record(
        longId='ab_calgary',
        property_type='LUXURY',
        version='202311',
        lang='en',
        user_prompt="No use prompt, this is a test",
        chatgpt_response="Calgary boasts an impressive skyline and lively cultural scene.",
    )
    self.archiver.add_record(
        longId='ab_calgary',
        property_type='LUXURY',
        version='202311',
        lang='en',
        user_prompt="Different prompt",
        chatgpt_response="Different response",
    )

    # Write the new record to the file by deleting the archiver instance
    del self.archiver

    # Create a new archiver instance
    self.archiver = ChatGPTRewriteArchiver(ArchiveStorageType.PLAIN_TEXT, file_path='test_archive.txt')

    # Check that get_record returns the new record
    record = self.archiver.get_record('ab_calgary', 'LUXURY', '202311', 'en', use_cache=True)
    self.assertEqual(record['user_prompt'], "Different prompt")
    self.assertEqual(record['chatgpt_response'], "Different response")

# TODO: Here are a few additional tests you could consider:

# Test adding and retrieving a record with special characters in the fields.
# Test adding and retrieving a record with very long field values.
# Test adding and retrieving a large number of records to test the performance of your implementation.
# Test the behavior when the file_path provided does not exist or is not accessible.
# Test the behavior when the file_path provided is not a text file.
# Test the behavior when the file_path provided is a text file but contains malformed records.