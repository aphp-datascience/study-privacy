args <- commandArgs(trailingOnly = TRUE) # k, overlap, step

# Import libraries
library(cluster)
library(TraMineR)

# Params
k = as.numeric(args[1]) # 3
overlap = as.logical(args[2]) # True
step =  as.numeric(args[3]) # step (inverse de K (papier)) # 8
# step = max(seqlength(seq_def))
seed = as.numeric(args[4])

cluster_traj <- function(shifted=FALSE, k=3, step=8, overlap=TRUE){
    
    if (shifted==TRUE){
        suffix = "_shifted"
        }else{
            suffix = ""
    }
    
    # Read data
    path_file = "/export/home/acohen/privacy/privacy/pipelines"
    df <- read.csv(file = paste(path_file,"../../data/patient_traj.csv", sep="/"), header = TRUE)
    # df <- df[1:100,]

    # Labels
    labels =  seqstatl(df$visit_source_value)

    # states
    states = 1 : length(labels)

    begin_col = paste("begin",suffix, sep="")
    end_col = paste("end",suffix, sep="")

    # Convert SPELL format to the STS state-sequence format 
    seqs <- seqformat(df, id = "person_id", begin = begin_col , end =end_col, status = "visit_source_value", from = "SPELL", to = "STS", process = FALSE, compress=FALSE, overwrite=FALSE)

    # Define seq object
    seq_def = seqdef(seqs,labels=labels, states=states, process=FALSE )

    # Compute distance matrix
    dist_matrix <- seqdist(seq_def, method = "CHI2",step = step , sm = "CONSTANT", with.missing = TRUE, overlap = overlap)

    # Clustering (PAM method)
    # https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/pam.html
    set.seed(seed)
    clusters = pam(dist_matrix, diss = TRUE,k = k, pamonce = 5)

    # Write 
    write.csv(clusters$clustering, paste(path_file,"/../../data/clustering",suffix,".csv", sep=""))
    write.csv(clusters$medoids, paste(path_file,"/../../data/medoids",suffix,".csv", sep="")) 
    }

suppressMessages({
cluster_traj(k=k, step=step, overlap=overlap, shifted=TRUE)
# cluster_traj(k=k, step=step, overlap=overlap, shifted=FALSE)
})