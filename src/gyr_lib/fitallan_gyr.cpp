#include "fitallan_gyr.h"

using namespace imu;

FitAllanGyr::FitAllanGyr( std::vector< double > sigma2s, std::vector< double > taus, double freq )
: C_Q_(0.0 )
, C_N_(0.0 )
, C_B_(0.0 )
, C_K_(0.0 )
, C_R_(0.0 )
, freq_(freq )
{
    if ( sigma2s.size( ) != taus.size( ) )
        std::cerr << "Error of point size" << std::endl;

    m_taus = taus;

    std::vector< double > init = initValue( sigma2s, taus );

    int num_samples = sigma2s.size( );
    double param[] = { init[0], init[1], init[2], init[3], init[4] };

    ceres::Problem problem;

    for ( int i = 0; i < num_samples; ++i )
    {
        // std::cout << "sigma " << i << " " << taus[i] << " " << sigma2s[i] << std::endl;

        ceres::CostFunction* f = new ceres::AutoDiffCostFunction< AllanSigmaError, 1, 5 >(
        new AllanSigmaError( sigma2s[i], taus[i] ) );

        problem.AddResidualBlock( f, NULL /* squared loss */, param );
    }
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.logging_type                 = ceres::SILENT;
    options.trust_region_strategy_type   = ceres::DOGLEG;
    options.max_num_iterations           = 1000;

    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );
    //        std::cout << summary.FullReport( ) << "\n";
    //    std::cout << "num_parameters " << summary.num_parameters << std::endl;

    C_Q_ = param[0];
    C_N_ = param[1];
    C_B_ = param[2];
    C_K_ = param[3];
    C_R_ = param[4];

    // std::cout << "C_Q_ " << C_Q_ //
    //           << " " << C_N_  //
    //           << " " << C_B_  //
    //           << " " << C_K_  //
    //           << " " << C_R_ << std::endl;

    std::cout << "================================================================== " << std::endl;
    std::cout << "### Continuous-time Allan variance coefficients" << std::endl;
    std::cout << "Quantization Noise (Q): " << getQ( ) << R"( rad)" << std::endl;
    std::cout << "White Rate Noise   (N): " << getN( ) << R"( rad / sqrt(s)        # Kalibr: \sigma_g, Gyroscope "white noise", gyroscope_noise_density)" << std::endl;
    std::cout << "Bias Instability   (B): " << getB( ) << R"( rad / s)" << std::endl;
    //std::cout << "Bias Instability   (B): " << getBiasInstability( ) << R"( rad / s, at )" << taus[findMinIndex( calcSimDeviation( taus ) )] << " s" << std::endl;
    std::cout << "Rate Random Walk   (K): " << getK( ) << R"( rad / (s * sqrt(s))  # Kalibr: \sigma_{bg}, Gyroscope "random walk", gyroscope_random_walk)" << std::endl;
    std::cout << "Angle Rate Ramp    (R): " << getR( ) << R"( rad / s^2)" << std::endl;
    std::cout << "================================================================== " << std::endl;
}

std::vector< double >
FitAllanGyr::initValue( std::vector< double > sigma2s, std::vector< double > taus )
{
    if ( sigma2s.size( ) != taus.size( ) )
        std::cout << " Error with data size!!!" << std::endl;

    Eigen::MatrixXd Y( sigma2s.size( ), 1 );

    for ( unsigned int index = 0; index < sigma2s.size( ); ++index )
    {
        Y( index, 0 ) = sqrt( sigma2s[index] );
    }
    //        std::cout << "Y " << Y << std::endl;

    int m_order = 2;

    Eigen::MatrixXd B( 2 * m_order + 1, 1 );
    B.setZero( );

    Eigen::MatrixXd F( taus.size( ), 2 * m_order + 1 );
    F.setZero( );

    for ( unsigned int index = 0; index < taus.size( ); ++index )
        for ( int order_index = 0; order_index < 2 * m_order + 1; ++order_index )
        {
            int kk                  = order_index - m_order;
            F( index, order_index ) = pow( sqrt( taus[index] ), kk );
        }
    //        std::cout << "F " << F << std::endl;

    Eigen::MatrixXd A = F.transpose( ) * F;
    B                 = F.transpose( ) * Y;
    //        std::cout << "B " << B << std::endl;
    //        std::cout << "A " << A << std::endl;

    Eigen::MatrixXd C = A.inverse( ) * B;
    // std::cout << "C " << C.transpose( ) << std::endl;

    std::vector< double > init;
    for ( int index = 0; index < 2 * m_order + 1; ++index )
        init.push_back( std::abs( C( index, 0 ) ) );

    return init;
}

std::vector< double >
FitAllanGyr::calcSimDeviation( const std::vector< double > taus ) const
{
    std::vector< double > des;
    for ( auto& tau : taus )
        des.push_back( sqrt( calcSigma2(C_Q_, C_N_, C_B_, C_K_, C_R_, tau ) ) );
    return des;
}

double
FitAllanGyr::getBiasInstability( ) const
{
    return findMinNum( calcSimDeviation( m_taus ) );
}

double
FitAllanGyr::findMinNum( const std::vector< double > num ) const
{
    double min = 1000.0;
    for ( unsigned int index = 0; index < num.size( ); ++index )
        min = min < num[index] ? min : num[index];
    return min;
}

int
FitAllanGyr::findMinIndex( std::vector< double > num )
{
    double min    = 1000.0;
    int min_index = 0;
    for ( unsigned int index = 0; index < num.size( ); ++index )
    {
        min_index = min < num[index] ? min_index : index;
        min       = min < num[index] ? min : num[index];
    }
    return min_index;
}

double
FitAllanGyr::calcSigma2(double C_Q, double C_N, double C_B, double C_K, double C_R, double tau ) const
{
    // clang-format off
  return  C_Q * C_Q / ( tau * tau )
      + C_N * C_N / tau
      + C_B * C_B
      + C_K * C_K * tau
      + C_R * C_R * tau * tau;
    // clang-format on
}

double
FitAllanGyr::getQ( ) const
{
    return sqrt(C_Q_ * C_Q_ ) / ( sqrt(3.0 ) );
}

double
FitAllanGyr::getN( ) const
{
    return sqrt(C_N_ * C_N_ );
}

double
FitAllanGyr::getB( ) const
{
    return sqrt(C_B_ * C_B_ * M_PI / (2 * log(2)));
}

double
FitAllanGyr::getK( ) const
{
    return sqrt( 3.0 * C_K_ * C_K_ );
}

double
FitAllanGyr::getR( ) const
{
    return sqrt( 2.0 * C_R_ * C_R_ );
}
