import os
import glob
import json
import argparse
import random
import shutil
from typing import List
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment


def convert_sox_audiofile(orig_file, dest_file):
    command = f"sox -v 0.99 -V1 {orig_file} -r 8000 -c 1 -b 16 {dest_file}"
    out = os.system(command)

def convert_folder(folder, extension='*.wav'):
    total_duration = 0
    convertion_folder = folder+'/16khz/'
    os.makedirs(convertion_folder, exist_ok = True)
    # extension = '*.wav' if 'wav' in folder else '*.flac'
    for file in glob.glob(os.path.join(folder, extension)):
        shutil.move(file, convertion_folder)
    for file in sorted(glob.glob(convertion_folder + extension)):
        filename = os.path.basename(file)
        # dest_file = os.path.join(folder, filename)
        dest_file = os.path.splitext(os.path.join(folder, filename))[0] + '.wav'
        convert_sox_audiofile(file, dest_file)
        duration = get_duration(dest_file)
        total_duration = total_duration + duration
    shutil.rmtree(convertion_folder)
    return total_duration

def parallelize_convert_folder(folder, extension='*.wav'):
    audios = []
    convertio_folder = folder+'/16khz/'
    os.makedirs(convertion_folder, exist_ok = True)
    # extension = '*.wav' if 'wav' in folder else '*.flac'
    for file in glob.glob(os.path.join(folder, extension)):
        shutil.move(file, convertion_folder)
    for file in sorted(glob.glob(convertion_folder + extension)):
        filename = os.path.basename(file)
        # dest_file = os.path.join(folder, filename)
        dest_file = os.path.splitext(os.path.join(folder, filename))[0] + '.wav'
        audios.append(tuple((file, dest_file)))
    with Pool(cpu_count()-1) as p:
      p.starmap(convert_sox_audiofile, audios)
    shutil.rmtree(convertion_folder)


def get_duration(file):
    try:
        sound = AudioSegment.from_file(file)
        return int(sound.duration_seconds * 1000)
    except Exception as Ex:
        print(f'Error with {file}, error: {Ex}')


def read_prompt_file(speaker_directory) -> List[str]:
    """
    :param speaker_directory: a directory containing the transcriptions for the audio files
    :return: a list containing the transcription for each audio file
    """
    files = os.listdir(os.path.join(speaker_directory, 'etc'))
    if 'prompts.txt' in files:
        with open(os.path.join(speaker_directory, 'etc', 'prompts.txt')) as file:
            return file.readlines()
    elif 'PROMPTS' in files:
        with open(os.path.join(speaker_directory, 'etc', 'PROMPTS')) as file:
            return file.readlines()
    # except FileNotFoundError as ex:
    else:
        print(os.listdir(os.path.abspath(speaker_directory)))
        print(os.listdir(os.path.abspath(os.path.join(speaker_directory, 'etc'))))
        raise FileNotFoundError('"%s" has no PROMTS file' % os.path.abspath(speaker_directory))


def generate_json_file(source: str, destination: str):
    """
    :param source:
    :param destination:
    :return:
    """
    if not os.path.isdir(source):
        raise FileNotFoundError('The corpus directory "%s" does not exist' % os.path.abspath(source))

    speaker_directories = os.listdir(source)
    data = []
    total_duration = 0

    for i, speaker_directory in enumerate(tqdm(speaker_directories)):
        # print('Processing folder %s / %s' % (i + 1, len(speaker_directories)))

        # get the prompt file from the speaker directory
        try:
            prompt_file = read_prompt_file(os.path.join(source, speaker_directory))
        except FileNotFoundError as ex:
            print(ex)
            continue

        folders = os.listdir(os.path.join(source, speaker_directory))
        if 'wav' in folders:
            _ = parallelize_convert_folder(os.path.join(source, speaker_directory, '*.wav'))
        else:
            _ = parallelize_convert_folder(os.path.join(source, speaker_directory, '*.flac'))
        for row in prompt_file:
            row = row.strip()
            if row != '' and len(row.split(' '))>2:
                try:
                    # recreate the path to the audio file
                    path = os.path.basename(row.split(' ')[0])
                    if 'wav' in folders:
                        # path = path.replace('/mfc/', '/wav/')
                        path = os.path.join(source, speaker_directory, 'wav', path)
                        path += '.wav'
                    else:
                        # path = path.replace('/mfc/', '/flac/')
                        path = os.path.join(source, speaker_directory, 'flac', path)
                        path += '.flac'
                    # path = os.path.join(source, path)
                    path = os.path.abspath(path)

                    # get transcription from prompt file
                    transcription = row.split(' ')[1:]
                    transcription = ' '.join(transcription).replace('\n', '').lower()
                    transcription = transcription.replace('-', '')

                    # determine the size of audio file
                    # size = os.path.getsize(path)
                    duration = get_duration(path)
                    total_duration += duration
                    
                    fragment_id = speaker_directory + '_' + os.path.splitext(os.path.basename(path))[0]
                    speaker_id = speaker_directory + '_spk'

                    data.append({
                        'audio_id': speaker_directory,
                        'side': random.choice(['client', 'operator']),
                        'fragment_id': fragment_id,
                        'duration': duration,
                        'transcript': transcription,
                        'speaker-id': speaker_id,
                        'wav_filename': os.path.join('01_Data_Transform', path.split('voxforge-corpus/')[-1]),
                        # 'size': size
                    })

                except Exception as ex:
                    print(f"{path}, {speaker_directory}, {ex}")

    # save training data to file
    with open(destination, 'w') as outfile:
        json.dump(data, outfile)
    print(f"Total duration: {total_duration/3.6e6}")
    df = pd.DataFrame(data)
    df.to_csv('/content/voxforge.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tool for preparing training data from the Voxforge corpus")
    parser.add_argument('source', help='directory of the corpus')
    parser.add_argument('destination', help='path of the new (json) file containing the training data')
    args = parser.parse_args()

    generate_json_file(args.source, args.destination)
