import wandb

def wandb_init(args, verbose = False):
    """
    args : args to log
        args.session_name
        args.wandb_id
        args.run_name
        args.project_name
        
        args.exp_id
        args.exp_name
        args.exp_description
        log_metrics: Metrics to log
    """
    log_metrics = []
    verbose=False
 
    # if wandb.run is not None:
    #     print(f" End in-flight wandb run . . .")
    #     wandb.finish()
    # else:
    #     print(f" Initiate new W&B job run")
    settings = wandb.Settings(
        show_errors=True,  # Show error messages in the W&B App
        silent=False,      # Disable all W&B console output
        show_warnings=True, # Show warning messages in the W&B App
        show_info=True, 
        console = "redirect",
        notebook_name = args.session_name
    )    
    if not hasattr(args,"wandb_id"):
        # opt['exp_id'] = wandb.util.generate_id()
        # print_dbg(f"{opt['exp_id']}, {opt['exp_name']}, {opt['project_name']}", verbose) 
        wandb_run = wandb.init(project=args.project_name, 
                            entity="kbardool", 
                            name = args.run_name,
                            resume="never" ,
                            settings = settings)
        args.wandb_id = wandb_run.id
    else:
        wandb_run = wandb.init(project=args.project_name, 
                            entity="kbardool", 
                            id = args.wandb_id, 
                            resume="must", 
                            settings = settings)
        args.wandb_id = wandb_run.id
        if wandb_run.project != '':
            args.project_name = wandb_run.project
        else:
            wandb_run.project = args.project_name 
            
        if wandb_run.name != '':
            args.exp_name = wandb_run.name
        else:
            wandb_run.name = args.exp_name 
            
        if wandb_run.notes != '':
            args.exp_description = wandb_run.notes
        else:
            wandb_run.notes = args.exp_description 
        
    wandb.config.update(args,allow_val_change= True)
    
    for metric in log_metrics:
        wandb.define_metric(metric, summary="last")
    # assert wandb.run is None, "Run is still running"
    if verbose:
        print(f" WandB Initialization -----------------------------------------------------------\n"
              f" PROJECT NAME: {wandb_run.project}\n"
              f" RUN ID      : {wandb_run.id} \n"
              f" RUN NAME    : {wandb_run.name}\n"     
              f" RUN NOTES   : {wandb_run.notes}\n"     
              f" --------------------------------------------------------------------------------")
    return wandb_run
 

def wandb_watch(item = None, criterion=None, log = 'all', log_freq = 1000, log_graph = True):
    """
    Note: Increasing the log frequency can result in longer run times
    """
    if item is not None:
        wandb.watch(item,
                    criterion=criterion,
                    log = log,            ###     Optional[Literal['gradients', 'parameters', 'all']] = "gradients",
                    log_freq = log_freq,
                    log_graph = log_graph
                   )        
    return
