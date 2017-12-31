function a = feedforwardKeras2D(image,Noise_var)
    thingSpeakURL = 'http://localhost:5000/estimate_channel_2D';
    addpath('matlab-json')
    json.startup
    real_image = real(image);
    img_image = imag(image);
    channel_image = zeros(size(image,1), size(image,2), 2);
    channel_image(:, :, 1) = real_image;
    channel_image(:, :, 2) = img_image;
    
    X = struct('image', channel_image, 'Noise_var', Noise_var);
    data =json.dump(X);

    options = weboptions('MediaType','application/json');
    response = webwrite(thingSpeakURL,data,options);
    channel_response = json.load(response);
    channel_response = permute(channel_response, [3, 1,2]);
    a = channel_response(:, :, 1) + 1i*channel_response(:,:,2);
end

