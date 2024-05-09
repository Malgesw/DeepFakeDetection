import os

def check_corrupted_images(folder):
    corrupted_folders = []
    for folder_name in os.listdir(folder):
        folder_path = os.path.join(folder, folder_name)
        if os.path.isdir(folder_path):
            images_corrupted = False
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if os.path.isfile(image_path):
                    try:
                        with open(image_path, 'rb') as image_file:
                            image_data = image_file.read()
                            if len(image_data) == 0:
                                images_corrupted = True
                                break
                    except (OSError, IOError):
                        images_corrupted = True
                        break
            if images_corrupted:
                corrupted_folders.append(os.path.basename(folder_path))
            check_corrupted_images(folder_path)

    corrupted_folders.sort()
    for corrupted_folder in corrupted_folders:
        print(corrupted_folder)

# Inserisci il percorso della cartella principale
main_folder = '/home/kaneki/Scaricati/FF++-20240507T201347Z-001/FF++/original_sequences'

check_corrupted_images(main_folder)