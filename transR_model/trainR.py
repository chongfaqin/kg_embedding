import config
import models
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
#Input training files from benchmarks/FB15K/ folder.
con = config.Config()
#True: Input test files from the same folder.
con.set_in_path("./trainData/")
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(8)
con.set_train_times(4000)
con.set_nbatches(64)
con.set_alpha(0.01)
con.set_margin(0.6)
con.set_bern(0)
#con.set_dimension(64)
#To set the dimensions of the entities
con.set_ent_dimension(32)
#To set the dimensions of the relations
con.set_rel_dimension(16)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("Adam")

#Models will be exported via tf.Saver() automatically.
con.set_export_files("./res/model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.TransR)
#Train the model.
con.run()
#To test models after training needs "set_test_flag(True)".
con.test()
