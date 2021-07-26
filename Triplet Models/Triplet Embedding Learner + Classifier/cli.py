import os
import sys
from time import time

import Models
import utils as u
from Snapshot import capture_snapshot
from MakeData import make_data
from Train import train_classifier, train_embedder
from RTApp import realtime

# ******************************************************************************************************************** #

def app():    
    args_1 = "--num-samples"
    args_2 = "--embed"
    args_3 = "--e-epochs"
    args_4 = "--c-epochs"
    args_5 = "--lower"
    args_6 = "--upper"
    args_7 = "--id"

    if args_1 in sys.argv:
        u.num_samples = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv:
        u.embed_layer_size = int(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv:
        u.e_epochs = int(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv:
        u.c_epochs = int(sys.argv[sys.argv.index(args_4) + 1])

    if args_5 in sys.argv:
        u.lower_bound_confidence = float(sys.argv[sys.argv.index(args_5) + 1])        
    if args_6 in sys.argv:
        u.upper_bound_confidence = float(sys.argv[sys.argv.index(args_6) + 1]) 

    if args_7 in sys.argv:
        u.device_id = int(sys.argv[sys.argv.index(args_7) + 1])

    u.breaker()
    u.myprint("\t    --- Application Start ---", color="green")
    
    while True:
        u.breaker()
        ch = input("1. Add Object\n2. Retrain\n3. Application\n4. Exit\n\nEnter Choice : ")

        if ch == "1":
            u.breaker()
            part_name = input("Enter part name : ")
            capture_snapshot(device_id=u.device_id, part_name=part_name, roi_extractor=Models.roi_extractor)

            u.breaker()
            u.myprint("Generating Feature Vector Data ...", "green")
            start_time = time()
            make_data(part_name=part_name, cls="Positive", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            make_data(part_name=part_name, cls="Negative", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            u.myprint("\nTime Taken [{}] : {:.2f} minutes".format(2*u.num_samples, (time()-start_time)/60), "green")
            
            embedder, batch_size, e_lr, e_wd = Models.build_embedder(embed=u.embed_layer_size)
            checkpoint_path = train_embedder(part_name=part_name, model=embedder, epochs=u.e_epochs, lr=e_lr, wd=e_wd, batch_size=batch_size, fea_extractor=Models.fea_extractor)
            classifier, batch_size, c_lr, c_wd = Models.build_classifier(embedding_net=embedder, path=checkpoint_path, embed=u.embed_layer_size)
            train_classifier(part_name=part_name, model=classifier, epochs=u.c_epochs, lr=c_lr, wd=c_wd, batch_size=batch_size, fea_extractor=Models.fea_extractor)
            realtime(device_id=u.device_id, part_name=part_name, model=classifier, save=False, show_prob=True, fea_extractor=Models.fea_extractor)
        
        elif ch == "2":
            u.breaker()
            part_name = input("Enter part name : ")

            u.breaker()
            u.myprint("Generating Feature Vector Data ...", "green")
            start_time = time()
            make_data(part_name=part_name, cls="Positive", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            make_data(part_name=part_name, cls="Negative", num_samples=u.num_samples, fea_extractor=Models.fea_extractor, roi_extractor=Models.roi_extractor)
            u.myprint("\nTime Taken [{}] : {:.2f} minutes".format(2*u.num_samples, (time()-start_time)/60), "green")

            embedder, batch_size, e_lr, e_wd = Models.build_embedder(embed=u.embed_layer_size)
            checkpoint_path = train_embedder(part_name=part_name, model=embedder, epochs=u.e_epochs, lr=e_lr, wd=e_wd, batch_size=batch_size, fea_extractor=Models.fea_extractor)
            classifier, batch_size, c_lr, c_wd = Models.build_classifier(embedding_net=embedder, path=checkpoint_path, embed=u.embed_layer_size)
            train_classifier(part_name=part_name, model=classifier, epochs=u.c_epochs, lr=c_lr, wd=c_wd, batch_size=batch_size, fea_extractor=Models.fea_extractor)
        
        elif ch == "3":
            u.breaker()
            part_name = input("Enter part name : ")

            embedder, _, _, _ = Models.build_embedder(embed=u.embed_layer_size)
            checkpoint_path = os.path.join(os.path.join(u.DATASET_PATH, part_name), "Checkpoints")
            model, _, _, _ = Models.build_classifier(embedding_net=embedder, path=checkpoint_path, embed=u.embed_layer_size)
            realtime(device_id=u.device_id, part_name=part_name, model=model, save=False, show_prob=True, fea_extractor=Models.fea_extractor)

        elif ch == "4":
            break
            
        else:
            u.myprint("\nINVALID CHOICE !!!", "red")
    
    u.breaker()
    u.myprint("\t    --- Application End ---", color="green")
    u.breaker()

# ******************************************************************************************************************** #
