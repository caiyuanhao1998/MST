def set_template(args):
    # Set the templates here
    if args.template.find('mst') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Phi'

    if args.template.find('gap_net') >= 0 or args.template.find('admm_net') >= 0 or args.template.find('dnu')>= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi_PhiPhiT'

    if args.template.find('tsa_net') >= 0:
        args.input_setting = 'HM'
        args.input_mask = None

    if args.template.find('hdnet') >= 0:
        args.input_setting = 'H'
        args.input_mask = None

    if args.template.find('dgsmp') >= 0:
        args.input_setting = 'Y'
        args.input_mask = None

    if args.template.find('birnat') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi'

    if args.template.find('mst_plus_plus') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'

    if args.template.find('lambda_net') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
