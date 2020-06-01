def init_tf(config_dict=dict()):
    if tf.get_default_session() is None:
        tf.set_random_seed(np.random.randint(1 << 31))
        create_session(config_dict, force_as_default=True)


def init_uninited_vars(vars=None):
    if vars is None: 
        vars = tf.global_variables()
        test_vars = []
        test_ops = []
        with tf.control_dependencies(None): # ignore surrounding control_dependencies
            for var in vars:
                assert is_tf_expression(var)
                try:
                    tf.get_default_graph().get_tensor_by_name(var.name.replace(':0', '/IsVariableInitialized:0'))
                except KeyError:
                    # Op does not exist => variable may be uninitialized.
                    test_vars.append(var)
                    with absolute_name_scope(var.name.split(':')[0]):
                        test_ops.append(tf.is_variable_initialized(var))
        init_vars = [var for var, inited in zip(test_vars, run(test_ops)) if not inited]
        run([var.initializer for var in init_vars])

