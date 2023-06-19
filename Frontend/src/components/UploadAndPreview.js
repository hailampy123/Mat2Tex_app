import axios from 'axios';
import React from 'react';
import { useState } from 'react';
import { InboxOutlined, SnippetsOutlined } from '@ant-design/icons';
import { message, Upload, Image, Row, Col, Button, Space } from 'antd';
import '../css/UploadAndPreview.css'
import { AppContext } from '../Context';
const { Dragger } = Upload;


const props = {
  name: 'file',
  multiple: false,
  onChange(info) {
    console.log(info)
    const { status } = info.file;
    if (status === 'done') {
      message.success(`${info.file.name} file uploaded successfully.`);
    }
    else if (status === 'error') {
      message.error(`${info.file.name} file upload failed.`);
    }
  },
  onDrop(e) {
    console.log('Dropped files', e.dataTransfer.files);
  },
};

const UploadAndPreview = () => {
	const {appPhase, imageDisplay, prediction, loading, result} = React.useContext(AppContext)
	const [phase, setPhase] = appPhase
	const [imageUrl, setImageUrl] = imageDisplay;
	const [bboxes, setBboxes] = prediction;
	const [isLoading, setIsLoading] = loading;

	//upload: upload / preview
	const [step, setStep] = useState('upload');

	const uploadImage = async options => {
		const { onSuccess, onError, file } = options;
		// console.log(typeof(file))
		// setTmpFile(file)
		var reader = new FileReader();
		reader.onload = function () {
			var thisImage = reader.result;
			localStorage.setItem("imgData", thisImage);
		};
		reader.readAsDataURL(file)
		// localStorage.setItem('imgData', file)
		const fmData = new FormData();
		const config = {
			headers: { "content-type": "multipart/form-data" },
		};
		fmData.append("image", file);
		try {
			const res = await axios.post(
				"https://api.imgbb.com/1/upload?key=e99170ca7ac760be3b92a14cebc3e1ee",
				// "",
				fmData,
				config
			);

			onSuccess("Ok");
			setStep('review');
			const image = {
				width:res.data.data.width,
				height:res.data.data.height,
				url:res.data.data.url,
				delurl:res.data.data.delete_url
			}
			console.log(image)
			setImageUrl(image);

		} catch (err) {
			console.log("Error: ", err);
			onError({ err });
		}
	};

	const handlePaste = async (e) => {
		if (e.clipboardData.files.length) {
			const fileObject = e.clipboardData.files[0];
			const file = {
				getRawFile: () => fileObject,
				name: fileObject.name,
				size: fileObject.size,
				type: fileObject.type,
				uid: fileObject.uid,
			};
			console.log(file.getRawFile())

			// console.log(createURLfile)


			const fmData = new FormData();
			const config = {
				headers: { "content-type": "multipart/form-data" },
			};
			fmData.append("image", file.getRawFile());
			try {
				const res = await axios.post(
					"https://api.imgbb.com/1/upload?key=e99170ca7ac760be3b92a14cebc3e1ee",
					// "",
					fmData,
					config
				);

				// onSuccess("Ok");
				setStep('review');
				console.log(res)
				message.success("Image uploaded successfully")
				const image={
					width:res.data.data.width,
					height:res.data.data.height,
					url:res.data.data.url
				}
				setImageUrl(image);
			} catch (err) {
				console.log("Error: ", err);
				// onError({ err });
			}
		} else {
			message.error("No image data")
		}
	}

	function dataURLtoFile(dataurl, filename) {
		var arr = dataurl.split(','),
			mime = arr[0].match(/:(.*?);/)[1],
			bstr = atob(arr[arr.length - 1]),
			n = bstr.length,
			u8arr = new Uint8Array(n);
		while(n--){
			u8arr[n] = bstr.charCodeAt(n);
		}
		return new File([u8arr], filename, {type:mime});
	}

	const handlePredict = async () => {
		try {
			setIsLoading(true)
			// console.log(typeof(tmpFile))
			const imgData = localStorage.getItem('imgData')
			var file = dataURLtoFile(imgData,'image.png');
			console.log(file)
			const fmData = new FormData();
			const config = {
				headers: { "content-type": "multipart/form-data" },
			};
			fmData.append("file", file);

			const res = await axios.post(
				"https://pacific-spire-54560.herokuapp.com/detect",
				// "",
				fmData,
				config
			).then(res => {
				setIsLoading(false)
				return res
			})
			console.log(res.data)
			console.log(res.data.predictions)
			const predictions = res.data.predictions
			const scale = imageUrl.width > 900 ? 900 / imageUrl.width : 1


			const returnBoxes = predictions.map((prediction, index) => {
				// const {coordinates, img_name, class: class_} = prediction
				// const [x1, y1, x2, y2] = coordinates
				const [x1, y1, x2, y2, confidence, className] = prediction
				return {
					id: index,
					x: x1*scale,
					y: y1*scale,
					width: (x2-x1)*scale,
					height: (y2-y1)*scale,
					confidence,
					label: (className === 'embedded') ? 1 : 0,
				}
			})

			console.log(returnBoxes)
			setBboxes(returnBoxes)

		} catch (err) {
			console.log('Error: ', err);
		}
	}

	const handleUploadAnother = async () => {
		console.log('upload another')
		try {
			console.log(imageUrl)
			const res = await axios.post(imageUrl.delurl)
			console.log(res)
			} catch (err) {
			console.log(err)
		}
	}

	const Preview = () => (
		<div>
				<Row justify="space-between" align="top" style={{width: '100%'}}>
					<Col span={20} style={{
						padding: '0 12px 0 12px',
						display: 'flex',
    					justifyContent: 'center',
						justifyContent: 'center',
						position: 'relative'
						}}
					>
						<img
							// preview = {false}
							src = {imageUrl.url}
							className='image-preview'
							style = {{
								display: 'block',
								width: '100%',
								maxWidth: '900px'
							}}
						/>
					</Col>
					<Col span={4}>
						<Space wrap direction='horizontal'>
						<Button onClick={() => {
							handlePredict();
							setStep('upload');
							setPhase('predict');
						}} type="primary"> Proceed to next step </Button>
						<Button onClick={() => {
							handleUploadAnother();
							setStep('upload');
						}}> Upload another </Button>
						</Space>
					</Col>
				</Row>
		</div>
	);

	const DragDrop = () => (
		<Space direction="vertical" size="middle" style={{ display: 'flex' }}>
		<Dragger
			customRequest = {uploadImage}
			accept='image/*'
			{...props}
		>
			<p className="ant-upload-drag-icon">
				<InboxOutlined />
			</p>
			<p className="ant-upload-text">Click or drag image to this area to upload</p>
			<p className="ant-upload-hint">
				Support for a single upload. Strictly prohibited from uploading company data or other
				banned files.
			</p>
		</Dragger>
		<div
			className="ant-upload-wrapper css-dev-only-do-not-override-w8mnev css-w8mnev css-htwhyh"
			onPaste = {(e) => {
				e.preventDefault()
				handlePaste(e)
			}}
		>
			<div
			className="ant-upload ant-upload-drag css-dev-only-do-not-override-w8mnev css-w8mnev css-htwhyh"
			>
				<span>
					<p className="ant-upload-drag-icon">
					<SnippetsOutlined />
					</p>
					<p className="ant-upload-text">Click this area and paste image to upload</p>
					<p className="ant-upload-hint">
						Support for a single upload. Strictly prohibited from uploading company data or other
						banned files.
					</p>
				</span>
			</div>
		</div>
	</Space>
	)

	return (<div> {step==='upload' ? <DragDrop/> : <Preview/>}
				</div>)
};

export default UploadAndPreview;