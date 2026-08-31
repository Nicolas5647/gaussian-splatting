# Pipeline de Gaussian Splatting

Voici les étapes à suivre pour utiliser les scripts. Consultez `README_3DGS.md` pour plus d'informations sur le code de `gaussian_splatting`.

## Etape 0 - lancer le docker
Dans la racine du repo **Gaussian Splatting** :

```bash
docker build -t gaussian_splatting .
```

Pour lancer le docker :

```bash
xhost +local:docker
```
Autoriser Docker à se connecter à l'écran de l'hôte

```bash
docker run --gpus all -it --rm \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v "<chemin/vers/dataset>:/app/data" \
    -v "<chemin/vers/output>:/app/output" \
    gaussian_splatting
```
Voici la commande pour quitter le conteneur : `exit`
L'explication des options :
* `--gpus all` : permet d'utiliser tous les GPUs disponibles
* `-it` : permet d'ouvrir un terminal
* `--rm` : supprime le conteneur après son exécution
* `--net=host` : partage directement la configuration et la pile réseau de la machine hôte avec le conteneur
* `-e DISPLAY=$DISPLAY`: transmet la variable d'environnement DISPLAY de votre machine hôte au conteneur. Cela indique aux applications graphiques du conteneur sur quel écran physique elles doivent envoyer l'affichage.
* `-v /tmp/.X11-unix:/tmp/.X11-unix:rw` : monte le dossier contenant les sockets Unix de communication X11 en mode lecture-écriture (rw). Cela permet aux applications graphiques du conteneur de communiquer avec l'hôte.
* `-v $HOME/.Xauthority:/root/.Xauthority:rw` : partage le fichier d'authentification .Xauthority de votre session utilisateur avec le dossier de l'utilisateur root dans le conteneur. Ce fichier contient des clés secrètes (cookies de sécurité) indispensables pour prouver au serveur d'affichage de l'hôte que le conteneur a légitimement le droit de s'y connecter.
* `-v "<chemin/vers/dataset>:/app/data"` : monte le dossier `dataset` du répertoire courant dans le dossier `/app/data` du conteneur.
* `-v "<chemin/vers/output>:/app/output"` : monte le dossier `output` du répertoire courant dans le dossier `/app/output` du conteneur.


## Étape 1 — Extraire les images de la vidéo

Extraire les images d'une vidéo avec **FFmpeg** :

```bash
ffmpeg -i <video> -qscale:v 1 -qmin 1 -vf fps=<fps> %04d.jpg
```

* `-i <video>` : spécifie le fichier vidéo source
* `-qscale:v 1` : contrôle la qualité de sortie des images (encodage Jpeg) :
  * 1 = meilleure qualité (fichier plus lourd)
  * 31 = qualité minimale (fichier plus léger)
* `-qmin 1` : définit la limite inférieure de l'échelle de qualité (garantit que la qualité ne descendra pas en dessous de ce seuil).
* `-vf "fps=<fps>"` : filtre vidéo permettant de définir la fréquence d'extraction des images :
  * Exemple : `fps=1` extrait une image par seconde
* `%04d.jpg` : définit le format de nommage des images de sortie avec un compteur sur 4 chiffres :
  * Génère des fichiers nommés `0001.jpg`, `0002.jpg`, `0003.jpg`, etc
  * Le format `%04d` est essentiel pour que le tri alphabétique des fichiers corresponde à l'ordre chronologique de la vidéo
  * On peut aussi mettre `input/%04d.jpg` pour avoir des images nommées `input/0001.jpg`, `input/0002.jpg`, etc

## Étape 2 — Créer un dataset COLMAP

Organiser les images ainsi :

```
scene_dataset/
 └── input/
     ├── 0001.jpg
     ├── 0002.jpg
     └── ...
```

## Étape 3 — Préparer les données pour la convertion en dataset exploitable

Pour convertis les images en dataset exploitable :

```bash
conda activate gaussian_splatting
python convert.py -s <chemin/vers/scene_dataset>
```

### Création des masques avec SAM v3

Pour créer le dossier des masques (**masks**) avec SAM v3, l'accès au modèle doit d'abord être autorisé sur Hugging Face ![lien](https://huggingface.co/facebook/sam3). 
Pour vous connecter, exécutez la commande suivante dans un terminal : `hf auth login`
Et entrez le tocken de connexion, ils sont disponibles dans settings > Access Tokens de votre profil Hugging Face.

Ensuite, lancez la commande suivante :
```bash
conda activate sam3
python segmentate.py -s <chemin/vers/scene_dataset> -v
```

* `-s <chemin/vers/scene_dataset>` : chemin vers le dossier de données
* `-sp <prompt>` : prompt pour SAM (si `-v` n'est pas utilisé)
* `-v` : Utilise le prompt visuel
* `--inv` : Permet d'appliquer un modele de diffusion pour compléter la zone masquée
* `-dp <prompt>` : prompt pour la diffusion (si `--inv` est utilisé)
* `-np <prompt>` : prompt negativ pour la diffusion (si `--inv` est utilisé)
* `--max-height <hauteur>` : Hauteur maximale de l'image (par défaut: 1080)
* `--max-width <largeur>` : Largeur maximale de l'image (par défaut: 1920)
* `--result <chemin/vers/output/>` : chemin vers le dossier de sortie pour enregistrer le resultat final (optionnel)

Exemple de prompt :
```bash
python segmentate.py -s ./data/tandt/truck/ -p "Blue truck with wooden side racks can see one part" --result ./data/tandt/truck/output/
python segmentate.py -s ./data/tandt/train/ -p "Green locomotive with orange stripes"  --result ./data/tandt/train/output
python segmentate.py -s ./data/trompe -v
```

## Étape 4 —  Reconstruction du point cloud

Pour avoir une interface graphique il faut lancer dans un autre terminal :
```bash
docker exec -it <id_container> bash
./SIBR_viewers/bin/SIBR_remoteGaussian_app
```
Pour avoir les ids des containers : `docker ps`

Pour lancer la reconstruction sans utiliser le masque:

```bash
conda activate gaussian_splatting
python train.py -s <chemin/vers/scene_dataset>
```

Si vous voulez utiliser les masques se trouvant dans le dossier `masks` :

```bash
python train.py -s <chemin/vers/scene_dataset> --use_mask
```

Pour utiliser Difix3D :

```bash
python train.py -s <chemin/vers/scene_dataset> --difix_iteration 30000 --difix_mode interp
```

Liste de quelques paramètres :
  * `-m` : chemin vers le dossier de sortie
  * `--use_mask` : active l'utilisation du masque
  * `--resolution` : facteur de division de la résolution des images d'entrée (1 pour la qualité maximale, 2 pour diviser par deux) (default : -1)
  * `--densify_grad_threshold` : seuil de la densification, c'est-à-dire la division ou de la duplication des splats (default : 0.0002)
  * ̀̀`--densify_from_iter` : itération de départ de la densification (default : 500)
  * `--densify_until_iter` : itération d'arrêt de la densification (default : 15000)
  * `--densification_interval` : intervalle entre deux densifications (default : 100)
  * `--iterations` : nombre d'itérations (default : 30000)
  * `--save_iterations` : sauvegarde en `ply` pour chaque itération (default : 7000 30000)
  * `--checkpoint_iterations` : sauvegarde en `pth` pour chaque itération (default : vide)
  * `--start_checkpoint` : charge le checkpoint à partir du fichier `pth` (default : vide)
  * `--difix_mode` : mode de correction Difix3D voici les options interp, spiral, ellipse, rotate (default : interp)
  * `--difix_iteration` : nombre d'itérations de la correction Difix3D (default : 0)
  * `--difix_step` : nombre d'itérations entre les corrections Difix3D (default : 2000)


## Étape 5 — Exporter le point cloud `.PLY`

Le fichier `.ply` est généré dans le dossier de sortie `./output/<run_id>/iteration_<num_iteration>/point_cloud.ply` ou dans le dossier de sortie spécifié par l'option `-m`.

# Experiments

Le script `experiments.py` permet de lancer plusieurs entraînements différents sur le même dataset. Il crée un dossier ayant le nom du dataset, et pour chaque entraînement, il crée un dossier avec le nom de l'entraînement et les paramètres utilisés. Il crée également le fichier `training_metrics.csv` dans `output` qui contient les temps de calcule pour chaque itération.

```bash
python experiments.py -s <chemin/vers/scene_dataset>
```

Paramètres :

* `-s <chemin/vers/scene_dataset>` : chemin vers le dossier de données
* `--use_mask` : active l'utilisation du masque
* `-r <resolution>` : facteur de division de la résolution des images d'entrée (ex: "1 4 8")
* `--densify_grad_threshold` : seuil de la densification, c'est-à-dire la division ou de la duplication des splats (default : 0.0002)
* `--densify_until_iter` : itération de départ de la densification (default : 500)

# Aruco

Voici un site pour generer des aruco : [https://chev.me/arucogen/](https://chev.me/arucogen/)
Ensuite, le script `aruco.py` permet de lancer l'estimation de la pose de l'aruco. Ils crée un dossier ayant le nom du dataset plus `_norm`, et copie le dossier `input`, `images` et `masks` du dataset original. Il crée également le fichier `sparse/0` contenant les poses normalisées. C'est le dossier `_norm` qui sera utilisé pour l'entraînement. Il faut avant activer l'environnement `aruco` avec `conda activate aruco`.

```bash
python aruco.py -s <chemin/vers/scene_dataset> -t <id_marker> -a <taille_aruco> -v
```

Paramètres :

* `-s <chemin/vers/scene_dataset>` : chemin vers le dossier de données
* `-t <id_marker>` : id du marker à utiliser
* `-a <taille_aruco>` : taille de l'aruco (en mètres)
* `-v` : affiche la vue 3D
