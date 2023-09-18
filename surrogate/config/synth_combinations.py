combinations_default_config = {
            "dim" : [range(2,25),range(25,50),range(50,100)],
            "n_clusters" : [range(2,9),range(10,17),range(18,22),range(23,35)],
            "n_samples" : [range(150,650,50),range(1500,3000,250),range(3000,5000,500)],
            "overlap_settings": [{"min_overlap":1e-6,"max_overlap":1e-5}],
            "aspect_ref" : [1.5, 5],
            "aspect_maxmin" : [1, 5],
            "radius_maxmin" : [1, 3, 5],
            "distributions" : [{'normal'}, {'exponential'}, {'gumbel'}],
            #"distribution_proportions" : ["low", "med", "high"],
            "imbalance_ratio" : [range(1,3)]
        }