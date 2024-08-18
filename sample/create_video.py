import sys
sys.path.append('.')
from Hopfield import HopfieldVis

if __name__ == '__main__':
    if len(sys.argv) <= 3:
        print('Usage:\n',
              f'{sys.argv[0]} IMAGE_FILE P M0\n\n',
              '- IMAGE_FILE: Image file that you want to memorize and restore\n',
              '- P: The number of image that the Hopfield network memorize (1 is the file that you provide and P - 1 is random-generated)\n'
              '- M0: Initial overlap',
              )
        sys.exit(0)
    image_name = sys.argv[1]
    p = int(sys.argv[2])
    m0 = float(sys.argv[3])
    print('Creating Hopfield network.')
    hopfield = HopfieldVis(image_name, p)
    print('Memorizing data.')
    hopfield.memorize()
    print('Recalling your image.')
    hopfield.recall(m0, 1e-4)
    print('Saving recalling video.')
    hopfield.save_video()
    print('Done.')

