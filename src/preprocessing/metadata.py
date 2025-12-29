import subprocess

def get_xml_metadata(imgPath):
  """
      get XML metadata
      This function retrieves metadata from an image file using ExifTool.
      Parameters:
      - imgPath: Path to the image file.
      Returns:
      - infoDict: Dictionary containing metadata tags and their values.
  """

  infoDict = {}
  exifToolPath = 'exiftool'
  ''' use Exif tool to get the metadata '''
  process = subprocess.Popen([exifToolPath,imgPath],stdout=subprocess.PIPE, stderr=subprocess.STDOUT,universal_newlines=True)
  ''' get the tags in dict '''
  for tag in process.stdout:
      line = tag.strip().split(':')
      infoDict[line[0].strip()] = line[-1].strip()
  return infoDict