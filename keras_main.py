import  ECS_Utility


if __name__ == '__main__':
    # 1. 读取配置文件
    config = ECS_Utility.read_config()
    # 2. 读取数据
    data = ECS_Utility.read_data(config)
    # 3. 预测
    result = ECS_Utility.predict_vm(config, data)
    # 4. 写入输出文件
    ECS_Utility.write_result(config, result)
    # 5. 输出结果
    ECS_Utility.print_result(result)
    # 6. 输出评估结果
    ECS_Utility.evaluate(config, data, result)

