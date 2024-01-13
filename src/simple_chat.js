import init, { Session, Input } from "@webonnx/wonnx-wasm";

			async function fetchBytes(url) {
				const reply = await fetch(url);
				const blob = await reply.arrayBuffer();
				const arr = new Uint8Array(blob);
				return arr;
			}

			function log(txt) {
				document.getElementById("log").append(txt);
			}

			async function run() {
				try {
					const [modelBytes, initResult] = await Promise.all([fetchBytes("/home/ultramarine/wonnx-experiments/torch-model.onnx"), init()])
					console.log("Initialized", { modelBytes, initResult, Session});
					const session = await Session.fromBytes(modelBytes);

					
					console.log({session});
					const input = new Input();
					input.insert("x", [13.0, -37.0]);
					console.time("inference");
					const result = await session.run(input);
					console.timeEnd("inference");
					console.log({result});
					log(JSON.stringify(Object.fromEntries(result), undefined, "\t"));
					session.free();
					input.free();
				}
				catch(e) {
					console.error(e, e.toString());
				}
			}

			run();