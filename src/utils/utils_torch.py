import os, sys, argparse
import torch
import numpy as np
import onnx, onnxruntime as ort


@torch.no_grad()
def pytorch2ONNX(model, shape, output_name, dynamic_batch=False, opset_v=11):
    input_names = ["input_0"]
    output_names = ["output_0"]

    dummy_input = torch.randn(shape).cuda()
    # model = torch.nn.Conv2d(3, 64, 3, 1, 1)
    model = model.cuda().eval()

    if not dynamic_batch:
        torch.onnx.export(
            model,
            dummy_input,
            output_name,
            export_params=True,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_v,
            do_constant_folding=True,
        )
    else:
        print("export onnx model with dynamic batch..")
        torch.onnx.export(
            model,
            dummy_input,
            output_name,
            export_params=True,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_v,
            do_constant_folding=True,
            dynamic_axes={
                "input_0": {0: "batch_size"},  # variable length axes
                "output_0": {0: "batch_size"},
            },
        )
    # check exported onnx file
    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    if opset_v <= 8:
        """add output shape 근데 이거 opset version <= 8 일때만 가능"""
        onnx.save(
            onnx.shape_inference.infer_shapes(onnx.load(output_name)), output_name
        )

    ort_session = ort.InferenceSession(output_name)
    ort_input_name = ort_session.get_inputs()[0].name
    print(f"onnx input name : {ort_input_name}")
    ort_input_shape = ort_session.get_inputs()[0].shape
    print(f"input shape : {ort_input_shape}")

    ort_output_name = ort_session.get_outputs()[0].name
    print(f"output name : {ort_output_name}")
    ort_output_shape = ort_session.get_outputs()[0].shape
    print(f"output shape : {ort_output_shape}")

    model = model.eval().cpu()
    dummy = torch.randn(shape)
    torch_out = model(dummy)

    # ONNX 런타임에서 계산된 결과값
    # ort_inputs = {ort_input_name: to_numpy(dummy_input)}
    # ort_outs = ort_session.run(None, ort_inputs)
    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    out_ort = ort_session.run(
        None,
        {
            ort_session.get_inputs()[0].name: to_numpy(dummy),
        },
    )[0]

    # print(torch_out.shape)
    # print(to_numpy(torch_out).shape)
    # diff = torch.eq(torch_out, torch.tensor(out_ort))
    # print(f'# of different value {(~diff).sum()}')
    # assert torch.all(diff)

    # 비교
    np.testing.assert_allclose(
        to_numpy(torch_out),
        out_ort,
        rtol=1e-03,
        atol=1e-04
        # atol=1e-03)
    )

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    onnx.checker.check_model(onnx.load(output_name))
    print("The model is checked!")


# if __name__ == "__main__":
#     sys.path.append(os.path.dirname(__file__) + "/../..")
#     # print(os.path.dirname(__file__) + '/../..')
#     parser = argparse.ArgumentParser(description="pytorch to onnx convertor")
#     parser.add_argument(
#         "--model", type=str, required=True, help="model name for converting"
#     )

#     parser.add_argument(
#         "--weight", type=str, required=True, help="weight file for model"
#     )

#     parser.add_argument(
#         "--shape", type=str, required=True, help="NxCxHxW, example : 1x3x256x256"
#     )

#     parser.add_argument(
#         "--output", type=str, required=True, help="output file name(onnx)"
#     )

#     parser.add_argument(
#         "--opset", type=str, required=True, help="opset version for converting"
#     )

#     parser.add_argument(
#         "--dynamic",
#         type=bool,
#         required=False,
#         default=False,
#         help="whether model has dynamic batch",
#     )

#     # parser.add_argument('--dynamicbatch',
#     #                     type=bool,
#     #                     required=True,
#     #                     help='whether dynamic batch is applied')

#     args = parser.parse_args()

#     shape = [int(shape_) for shape_ in args.shape.split("x")]

#     from models import build_model

#     model = build_model(args.model, 702, onnx=True)

#     model.load_state_dict(torch.load(args.weight))
#     pytorch2ONNX(model, shape, args.output, args.dynamic, int(args.opset))
