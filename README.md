## Introduction

Le projet de ce notebook a été réalisé dans le cadre de la [formation d'ingénieur machine learning proposé par Openclassrooms](https://openclassrooms.com/fr/paths/148-ingenieur-machine-learning).

Il portait sur la comparaison entres des modèles de computer vision  entraînés initialement et l'utilisation de transfer learning. 

La démarche a été réalisée de manière itérative par entraînements sucessifs de "nouveaux modèles":


1.  Modèle initial avec préprocessing seul
2.  Ajout d'optimisations : dropout, batchnormalization ...
3.  Implémentation de la data augmentation 
4.  Transfer learning avec optimisation et data augmentation 

Ces quatres étapes ont été réalisée sur la base d'un modèle VGG16 : 

*  Reconstitué par empilement de couches pour les trois premières étapes
*  Modèle pré-entrainé sur ImageNet pour la quatrième

Les performances du modèle VGG16 pré-entrainé a ensuite été comapré avec RESNET50 et Xception.

L'ensemble des travaux ont été menés à l'aide du dataset [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/). Ce dernier est constitué de 20 580 images de chiens triées en 120 classes relatives à leur race. 

Les entrainements ont été réalisés sur GPU à l'aide de Google Colab.

Le meileur modèle a été par la intégré dans un démonstrarteur développé à l'aide du framework [Streamlit](https://streamlit.io).

## Contenu du repositiry:

*  Un notebook d'entraînement des modèles
*  Une application streamlit 
*  Une présentation du projet

## Données: 
dataset [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)

## Mode d'emploi pour le lancement de l'application
1.  Installer des librairies utilisées pour les projet : 
```pip install -r requrements.txt```

2.  Lancer de l'application streamlit:
```streamlit run 6_02_app.py```

3.  Ouvrir uen fenêtre de navigateur avec l'URL ```http://localhost:8501```