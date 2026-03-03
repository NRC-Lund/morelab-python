import qtm

def apply_aim_model():
    title = "Select AIM Model to Apply"
    filters = ["AIM files (*.qam)"]
    multiselect = False
    directory = qtm.settings.directory.get_aim_directory()
    model = qtm.gui.dialog.show_open_file_dialog(title, filters, multiselect, directory)
    model_path = model[0]
    print(model_path)
    
    # Where to apply the AIM model? Frames? Unidentified markers?
    qtm.settings.processing.aim.set_model_is_applied("project", model_path, True)
    settings = qtm.settings.processing.aim.get_settings("project")
    settings['keep_existing_labels'] = True
    print(settings)
    qtm.processing.apply_aim(settings)